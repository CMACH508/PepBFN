import os
import sys
sys.path.append("/home/qianhao/peptide")
import pytorch_lightning as pl
import numpy as np
import torch
import argparse
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from tqdm import tqdm
from core.utils.geometry import get_chain_from_pdb, get_CA_dist, get_psi_chi, diff_ratio, get_seq 
from easydict import EasyDict


class EvalPep(Callback):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.pep_dir = self.cfg.accounting.generated_pep_dir
    
    def eval_metric(self):
        """
        Evaluate the metrics of the generated samples and save them.
        """
        pdb_ids = os.listdir(self.pep_dir)
        eval_res = {}
        for pdb_id in tqdm(pdb_ids, desc="Evaluating metrics"):
            gt_pdb_path = os.path.join(self.pep_dir, pdb_id, 'gt.pdb')
            gt_chain_id = pdb_id.split('_')[-3]  # Assuming the chain ID is the last character before the file extension
            gt_chain = get_chain_from_pdb(gt_pdb_path, gt_chain_id)
            # gt_psi, gt_phi = get_psi_chi(gt_pdb_path, gt_chain_id)
            eval_res[pdb_id] = {
                        'gt_CA_dist': [],
                        # 'gt_psi': gt_psi,
                        # 'gt_phi': gt_phi,
                        'sample_CA_dist': [],
                        # 'sample_psi': [],
                        # 'sample_phi': [],
                        'aar':[]
                    }
            eval_res[pdb_id]['gt_CA_dist'].append(get_CA_dist(gt_chain))
            for i in range(self.cfg.num_samples):
                pdb_id_sample = f"sample_{i}.pdb"
                pdb_i_path = os.path.join(self.pep_dir, pdb_id, pdb_id_sample)
                
                if not os.path.exists(pdb_i_path):
                    rank_zero_info(f"Sample {pdb_i_path} does not exist.")
                else:
                    pdb_i_chain = get_chain_from_pdb(pdb_i_path, gt_chain_id)
                    try:
                        eval_res[pdb_id]['sample_CA_dist'].append(get_CA_dist(pdb_i_chain))
                    except Exception as e:
                        rank_zero_info(f"Error checking peptide validity for {pdb_i_path}: {e}")
                    
                    # try:
                    #     sample_psi, sample_phi = get_psi_chi(pdb_i_path, gt_chain_id)
                    #     eval_res[pdb_id]['sample_psi'].append(sample_psi)
                    #     eval_res[pdb_id]['sample_phi'].append(sample_phi)
                    # except Exception as e:
                    #     rank_zero_info(f"Error calculating psi/phi for {pdb_i_path}: {e}")
                        
                    try:
                        # Calculate amino acid ratio (AAR)
                        aar = diff_ratio(get_seq(pdb_i_chain), get_seq(gt_chain))
                        eval_res[pdb_id]['aar'].append(aar)
                    except Exception as e:
                        rank_zero_info(f"Error calculating AAR for {pdb_i_path}: {e}")
        if 'generated_pep_packsc' in self.pep_dir:
            torch.save(eval_res, os.path.join(self.cfg.accounting.logdir, 'eval_other_metrics_sc.pt'))
        else:
            torch.save(eval_res, os.path.join(self.cfg.accounting.logdir, 'eval_other_metrics.pt'))
        
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when testing ends.
        """
        if trainer.global_rank == 0:
            self.eval_metric()
            
if __name__ == "__main__":
    # NOTE: This is a standalone script for side chain packing purposes.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='/home/qianhao/peptide/logs/qianhao_bfn_peptide/debug/transnorm5_seqs_rots_torus')
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sc_packing", action='store_true', help="Whether to use side chain packing.")
    _args = parser.parse_args()
    # Example usage
    from core.config.config import Config
    config_file = os.path.join(_args.root_dir, 'config.yaml')
    cfg = Config(config_file)
    if _args.sc_packing:
        cfg.accounting.generated_pep_dir = os.path.join(os.path.dirname(cfg.accounting.generated_pep_dir), 'generated_pep_packsc')
    
    cfg.num_samples = _args.num_samples
    eval_callback = EvalPep(cfg=cfg)
    eval_callback.on_test_end(EasyDict({"global_rank": 0}), None)  # Replace with actual trainer and module instances in practice