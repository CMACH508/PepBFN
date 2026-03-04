import os
import sys
sys.path.append("/home/qianhao/peptide")
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch.nn.functional as F
from tqdm.auto import tqdm
from core.utils.train import recursive_to
from core.modules.protein.writers import save_pdb
from core.models.torsion import full_atom_reconstruction, get_heavyatom_mask
from core.modules.common.geometry import construct_3d_basis
from core.modules.protein.constants import BBHeavyAtom
from core.dataset.so3_utils import rotmat_to_rotvec
import argparse
from easydict import EasyDict
class ConsPep(Callback):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.pep_dir = self.cfg.accounting.generated_pep_dir
        self.test_dir = self.cfg.accounting.test_outputs_dir
        # self.batch_pos = torch.tensor([])
    
    def save_samples_sc(self, samples, save_dir):
        # meta data
        batch = recursive_to(samples['batch'],'cpu')
        chain_id = [list(item) for item in zip(*batch['chain_id'])][0] # fix chain id in collate func
        icode = [' ' for _ in range(len(chain_id))] # batch icode have same problem
        id = batch['id'][0]
        # batch convert
        # aa=batch['aa] if only bb level
        samples['seqs'] = samples['seqs'].argmax(-1)
        pos_ha,_,_ = full_atom_reconstruction(R_bb=samples['rotmats'],t_bb=samples['trans'],angles=samples['angles'],aa=samples['seqs']) # (32,L,14,3), instead of 15, ignore OXT masked
        pos_ha = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.) # (32,L,A,3) pos14 A=14
        pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
        mask_new = get_heavyatom_mask(samples['seqs'])
        aa_new = samples['seqs']
        for i in range(self.cfg.num_samples):
            data_saved = {
                        'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                        'aa':aa_new[i], 'mask_heavyatom':mask_new[i], 'pos_heavyatom':pos_new[i],
                        }
            save_pdb(data_saved,path=os.path.join(save_dir,f'sample_{i}.pdb'))
        data_saved = {
                        'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                        'aa':batch['aa'][0], 'mask_heavyatom':batch['mask_heavyatom'][0], 'pos_heavyatom':batch['pos_heavyatom'][0],
                    }
        save_pdb(data_saved,path=os.path.join(save_dir,f'gt.pdb'))
        
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when testing ends.
        """
        if trainer.global_rank == 0:
            print("Saving PEP results...")
            names = [n.split('.')[0] for n in os.listdir(self.test_dir) if n.split('.')[1]=='pt']
            for name in tqdm(names):
                # name = '_'.join(name.split('_')[:2])
                pdb_dir = os.path.join(self.pep_dir, name)
                sample = torch.load(os.path.join(self.test_dir,f'{name}.pt'))
                os.makedirs(pdb_dir, exist_ok=True)
                self.save_samples_sc(sample[-1],pdb_dir)
                
            print(f"PEP results saved to {self.test_dir}")
            
            
if __name__ == "__main__":
    # NOTE: This is a standalone script for side chain packing purposes.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='/data2/qianhao/peptide/torus_1-error_portion')
    parser.add_argument("--num_samples", type=int, default=10)
    _args = parser.parse_args()
    # Example usage
    from core.config.config import Config
    config_file = os.path.join(_args.root_dir, 'config.yaml')
    cfg = Config(config_file)
    
    cfg.num_samples = _args.num_samples
    eval_callback = ConsPep(cfg=cfg)
    eval_callback.on_test_end(EasyDict({"global_rank": 0}), None)  # Replace with actual trainer and module instances in practice