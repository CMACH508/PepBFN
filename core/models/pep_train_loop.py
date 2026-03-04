import os
import torch
import pytorch_lightning as pl
from core.config.config import Config
from core.models.bfn_pep import BFNModel
from core.utils.train import get_optimizer, get_scheduler, sum_weighted_losses
from core.utils.data import repeat_batch
from core.dataset.pep_dataloader import ReferenceArray, RotReferenceArray
import time
class PepTrainLoop(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFNModel(self.cfg.dynamics)
        self.save_hyperparameters(self.cfg.todict())
        
    def forward(self):
        pass

    def on_fit_start(self):
        if self.cfg.dynamics.model.interpolant.sample_torus:
            # t1=time.time()
            self.dynamics.gmm_ref_array = ReferenceArray(self.cfg.data.gmm_mmap_path)
            # self.dynamics.rot_ref_array = RotReferenceArray(self.cfg.data.rot_mmap_path)
            # t2=time.time()
            # print(f'Loading reference array took {t2-t1} seconds')
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.dynamics(batch)
        
        loss = sum_weighted_losses(loss_dict, self.cfg.train.loss_weights)
        
        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'train/loss': loss.detach(),
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True
        )
        self.log_dict(
            {
                f'train/{k}': v.detach() for k, v in loss_dict.items()
            },
            on_step=True,
            on_epoch=False,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return None

        return loss

    def validation_step(self, batch, batch_idx):
        # eval with fixed sequence
        error = self.dynamics.fix_seq_sample(batch, num_steps=100, pos_norm=self.cfg.train.normalizer_dict.pos)
        
        self.log_dict(
            error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True
        )
        
        
        sum_batches, sum_loss, sum_loss_trans, sum_loss_rots, sum_loss_seqs, sum_loss_torsion, sum_loss_bb_atom, sum_loss_angle = 0, 0., 0., 0., 0., 0., 0., 0.
        # sample a random timestep for reconstruction loss computation
        num_graphs = batch['pos_heavyatom'].shape[0]
        for t in range(0, self.dynamics.discrete_steps, 10):
            sum_batches+=1
            t = torch.tensor(
                [t / float(self.cfg.dynamics.discrete_steps)], 
                device=batch['pos_heavyatom'].device
            ).repeat(num_graphs, 1)

            if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
                t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]

            # compute bfn loss
            loss_dict = self.dynamics(batch, t)
            loss = sum_weighted_losses(loss_dict, self.cfg.train.loss_weights)
            sum_loss += float(loss)
            sum_loss_trans += float(loss_dict['trans_loss'])
            sum_loss_rots += float(loss_dict['rot_loss'])
            sum_loss_seqs += float(loss_dict['seqs_loss'])
            sum_loss_torsion += float(loss_dict['torsion_loss'])
            sum_loss_bb_atom += float(loss_dict['bb_atom_loss'])
            sum_loss_angle += float(loss_dict['angle_loss'])
        
            
        recon_loss = {
            "val/recon_loss": sum_loss / sum_batches,
            "val/recon_loss_trans": sum_loss_trans / sum_batches,
            "val/recon_loss_rots": sum_loss_rots / sum_batches,
            "val/recon_loss_seqs": sum_loss_seqs / sum_batches,
            "val/recon_loss_torsion": sum_loss_torsion / sum_batches,
            "val/recon_loss_bb_atom": sum_loss_bb_atom / sum_batches,
            "val/recon_loss_angle": sum_loss_angle / sum_batches,
        }
        self.log_dict(
            recon_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True
        )
        return recon_loss["val/recon_loss"]
        
    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)
        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
        }
        
    def on_test_epoch_start(self):
        os.makedirs(self.cfg.accounting.test_outputs_dir, exist_ok=True)
        self.dic = {'id':[],'len':[],'tran':[],'aar':[],'rot':[],'trans_loss':[],'rot_loss':[]}
    
    def test_step(self, batch, batch_idx):
        batch_repeat = repeat_batch(batch, self.cfg.num_samples)
        traj_1 = self.dynamics.sample(batch_repeat,num_steps=self.cfg.sample_steps,pos_norm=self.cfg.train.normalizer_dict.pos)
        torch.save(traj_1,f'{self.cfg.accounting.test_outputs_dir}/{batch["id"][0]}_batchid_{batch_idx}.pt')