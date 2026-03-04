from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from Bio.PDB import PDBParser, is_aa
import numpy as np
from multiprocessing import Pool
import argparse
import glob
import os
import torch
def get_chain_dic(input_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)
    chain_dic = {}
    for model in structure:
        for chain in model:
            chain_dic[chain.id] = len([res for res in chain if is_aa(res) and res.has_id('CA')])

    return chain_dic

# 在子进程刚启动时初始化一次
def init_worker():
    init("-mute all")
    
def get_rosetta_score_base(pdb_path,chain_id):
    try:
        print(f'Processing Rosetta score for {pdb_path} with chain {chain_id}')
        pose = pose_from_pdb(pdb_path)
        chains = list(get_chain_dic(pdb_path).keys())
        chains.remove(chain_id)
        interface = f'{chain_id}_{"".join(chains)}'
        fast_relax = FastRelax() # cant be pickled
        scorefxn = get_fa_scorefxn()
        fast_relax.set_scorefxn(scorefxn)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        stabs,binds = [],[]
        for i in range(5):
            fast_relax.apply(pose)
            stab = scorefxn(pose)
            mover.apply(pose)
            bind = pose.scores['dG_separated']
            stabs.append(stab)
            binds.append(bind)
        return {'name':pdb_path,'stab':np.array(stabs).mean(),'bind':np.array(binds).mean()}
    except Exception as e:
        print(f"Error processing reference Rosetta score for {pdb_path}: {e}")
        return {'name':pdb_path,'stab':np.nan,'bind':np.nan}
                
def run_rosetta_batch(args, num_workers=32):
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.starmap(get_rosetta_score_base, args)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rosetta batch processing")
    parser.add_argument('--root_dir', help='PDB file paths', default='/home/qianhao/peptide/logs/qianhao_bfn_peptide/debug/transnorm5_seqs_rots')
    parser.add_argument('--num_workers', help='Number of worker processes', type=int, default=16)
    parser.add_argument('--rank', help='Rank of the process', type=int, default=0)
    parser.add_argument('--world_size', help='Total number of processes', type=int, default=11)
    parser.add_argument('--sc_packing', help='If perform side chain packing', action='store_true')
    args = parser.parse_args()
    # Batch processing example
    if args.sc_packing:
        pdb_paths = os.path.join(args.root_dir, 'generated_pep_packsc')
    else:
        pdb_paths = os.path.join(args.root_dir, 'generated_pep')
    pdbs = sorted(glob.glob(os.path.join(pdb_paths,'*','*.pdb')))[args.rank::args.world_size]
    chain_ids = [p.split('/')[-2].split('_')[-3] for p in pdbs]
    results = run_rosetta_batch(list(zip(pdbs, chain_ids)), num_workers=args.num_workers)
    if args.sc_packing:
        torch.save(results, os.path.join(os.path.dirname(pdb_paths), f'rosetta_results_{args.rank}_sc.pt'))
    else:
        torch.save(results, os.path.join(os.path.dirname(pdb_paths), f'rosetta_results_{args.rank}.pt'))