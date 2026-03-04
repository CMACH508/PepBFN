#!usr/bin/env python
import os
import uuid
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
from pyrosetta import *
import pyrosetta.rosetta as rosetta
from functools import partial
import time
import shutil
init(
    "-out:levels core.conformation.Conformation:error "
    "core.pack.pack_missing_sidechains:error "
    "core.pack.dunbrack.RotamerLibrary:error "
    "core.scoring.etable:error "
    "-packing:repack_only "
    "-ex1 -ex2 -ex3 -ex4 "
    "-multi_cool_annealer 5 "
    "-no_his_his_pairE "
    "-linmem_ig 1 "
    # "-mute all"
    # silent=True
)
from pyrosetta import init, Pose, get_fa_scorefxn, standard_packer_task, pose_from_file  # noqa
from pyrosetta.rosetta import core, protocols
import numpy as np
from multiprocessing import Pool
import glob

def packer_task(pdb_out, pdb_in, n_decoys=1):
    """
    Demonstrates the syntax necessary for basic usage of the PackerTask object
    performs demonstrative sidechain packing and selected design
    using  <pose>  and writes structures to PDB files if  <PDB_out>
    is True

    """
    pose = Pose()
    pose_from_file(pose, pdb_in)
    uid = "res_file" + str(uuid.uuid4())
    tmp = "./tmp/"
    os.makedirs(tmp, exist_ok=True)
    resfile = os.path.join(tmp, uid)
    pyrosetta.toolbox.generate_resfile.generate_resfile_from_pdb(
        pdb_in, resfile, pack=True, design=True, input_sc=False, freeze=[], specific={}
    )

    best_score, best_pose = 1e10, Pose()
    scorefxn = get_fa_scorefxn()
    for _ in range(n_decoys):
        test_pose = Pose()
        test_pose.assign(pose)
        # packer task
        pose_packer = standard_packer_task(test_pose)
        rosetta.core.pack.task.parse_resfile(test_pose, pose_packer, resfile)
        pose_packer.restrict_to_repacking()  # turns off design
        # pose_packer.or_include_current(True)  # considers original conformation
        packmover = protocols.minimization_packing.PackRotamersMover(scorefxn, pose_packer)

        scorefxn(pose)  # to prevent verbose output on the next line
        # print("\nPre packing score:", scorefxn(test_pose))
        packmover.apply(test_pose)
        # print("Post packing score:", scorefxn(test_pose))
        score = scorefxn(test_pose)
        if score < best_score:
            best_score = score
            best_pose = best_pose.assign(test_pose)

    best_pose.dump_pdb(pdb_out)



def safe_run(arg):
    try:
        packer_task(*arg, n_decoys=args.n_decoys)
    except Exception as e:
        print(f"Error processing {arg[1]}: {e}")
        print(f"skipping {arg}")
        
if __name__ == "__main__":
    parser = ArgumentParser(
        description=" Rosetta Pack",  # noqa
        epilog="run rosetta fixed backbone packing protocl",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root_dir", default="/home/qianhao/peptide/logs/qianhao_bfn_peptide/debug/transnorm5_seqs_rots_torus", help="root directory for data loading", type=str)
    parser.add_argument("--threads", type=int, default=11)
    parser.add_argument("--n_decoys", type=int, default=1)
    args = parser.parse_args()
    
    model_list = glob.glob(os.path.join(args.root_dir, "generated_pep", "*", "*.pdb"))
    out_root = os.path.join(args.root_dir, 'generated_pep_packsc')
    os.makedirs(out_root, exist_ok=True)
    arg_list = []
    for i, pdb_in in enumerate(model_list):
        pdb = os.path.basename(pdb_in)
        dir_name = os.path.basename(os.path.dirname(pdb_in))
        out_dir = os.path.join(out_root, dir_name)
        os.makedirs(out_dir, exist_ok=True)
        pdb_out = os.path.join(out_dir, pdb)
        if pdb == "gt.pdb":
            shutil.copyfile(pdb_in, pdb_out)
        else:
            arg_list.append((pdb_out, pdb_in))

    with Pool(args.threads) as p:
        p.map(safe_run, arg_list)