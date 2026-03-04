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
from core.utils.geometry import get_chain_from_pdb, get_bind_ratio, get_ss, \
    get_rmsd, get_peptide_valid, compute_diversity_avg, get_seq, get_novel    
from easydict import EasyDict
class EvalPep(Callback):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.pep_dir = self.cfg.accounting.generated_pep_dir
        self.test_dir = self.cfg.accounting.test_outputs_dir
    
    def eval_metric(self):
        """
        Evaluate the metrics of the generated samples and save them.
        """
        pdb_ids = os.listdir(self.pep_dir)
        eval_res = {}
        for pdb_id in tqdm(pdb_ids, desc="Evaluating metrics"):
            pdb_i_chain_tmp, pdb_i_chain_seq_tmp, rossetta_path_tmp = [], [], []
            gt_pdb_path = os.path.join(self.pep_dir, pdb_id, 'gt.pdb')
            gt_chain_id = pdb_id.split('_')[-3]  # Assuming the chain ID is the last character before the file extension
            gt_chain = get_chain_from_pdb(gt_pdb_path, gt_chain_id)
            rossetta_path_tmp.append(gt_pdb_path)
            eval_res[pdb_id] = {
                        'valid': [],
                        'rmsd': [],
                        'ss_ratio': [],
                        'bind_ratio': [],
                        'novel': [],
                        # 'affinity': [],
                        # 'stability': [],
                        # 'ref_affinity': [],
                        # 'ref_stability': [],
                    }
            for i in range(self.cfg.num_samples):
                pdb_id_sample = f"sample_{i}.pdb"
                pdb_i_path = os.path.join(self.pep_dir, pdb_id, pdb_id_sample)
                
                if not os.path.exists(pdb_i_path):
                    rank_zero_info(f"Sample {pdb_i_path} does not exist.")
                else:
                    pdb_i_chain = get_chain_from_pdb(pdb_i_path, gt_chain_id)
                    pdb_i_chain_tmp.append(pdb_i_chain)
                    pdb_i_chain_seq_tmp.append(get_seq(pdb_i_chain))
                    rossetta_path_tmp.append(pdb_i_path)
                    try:
                        bind_ratio = get_bind_ratio(pdb_i_path,gt_pdb_path,gt_chain_id,gt_chain_id)
                    except Exception as e:
                        bind_ratio = np.nan
                        rank_zero_info(f"Error processing {pdb_i_path}: {e}")
                    
                    try:
                        ss_ratio = get_ss(pdb_i_path,gt_chain_id,gt_pdb_path,gt_chain_id)
                    except Exception as e:
                        ss_ratio = np.nan
                        rank_zero_info(f"Error processing secondary structure for {pdb_i_path}: {e}")

                    try:
                        rmsd = get_rmsd(pdb_i_chain,gt_chain)
                    except Exception as e:
                        rmsd = np.nan
                        rank_zero_info(f"Error processing RMSD for {pdb_i_path}: {e}")
                    
                    try:
                        valid = get_peptide_valid(pdb_i_chain)
                    except Exception as e:
                        valid = np.nan
                        rank_zero_info(f"Error checking peptide validity for {pdb_i_path}: {e}")
                    
                    try:
                        novel = get_novel(pdb_i_chain, gt_chain)
                    except Exception as e:
                        novel = np.nan
                        rank_zero_info(f"Error checking novelty for {pdb_i_path}: {e}")
                        
                    obj = {
                        'valid': valid,
                        'rmsd': rmsd,
                        'ss_ratio': ss_ratio,
                        'bind_ratio': bind_ratio,
                        'novel': novel,
                        # 'affinity': rosetta_res['bind'],
                        # 'stability': rosetta_res['stab'],
                        # 'ref_affinity': ref_rossetta_res['bind'],
                        # 'ref_stability': ref_rossetta_res['stab'],
                    }
                    for k, v in eval_res[pdb_id].items():
                        eval_res[pdb_id][k].append(obj[k])
            
            div = compute_diversity_avg(pdb_i_chain_tmp, pdb_i_chain_seq_tmp)  
            eval_res[pdb_id]['diversity'] = div
            
            # eval affinity
            # t1 = time.time()
            # ref_rossetta_res = run_rosetta_batch(list(zip(rossetta_path_tmp, [gt_chain_id]*len(rossetta_path_tmp))))
            # t2 = time.time()
            # rank_zero_info(f"Processed reference Rosetta score in {t2 - t1:.2f} seconds for {gt_pdb_path}")            
        if 'generated_pep_packsc' in self.pep_dir:
            torch.save(eval_res, os.path.join(self.cfg.accounting.logdir, 'eval_metrics_sc.pt'))
        else:
            torch.save(eval_res, os.path.join(self.cfg.accounting.logdir, 'eval_metrics.pt'))
        
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
    parser.add_argument("--sc_packing", action='store_true')
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