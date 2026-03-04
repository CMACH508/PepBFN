from Bio.PDB import PDBParser, Superimposer, is_aa, Select, NeighborSearch
import tmtools
import numpy as np
import mdtraj as md
from Bio.SeqUtils import seq1

import warnings
from Bio import BiopythonWarning, SeqIO

import difflib

# 忽略PDBConstructionWarning
warnings.filterwarnings('ignore', category=BiopythonWarning)

def get_chain_from_pdb(pdb_path, chain_id='A'):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_path)[0]
    for chain in structure:
        if chain.id == chain_id:
            # print(len(chain))
            return chain
    return None

def get_CA_dist(chain):
    """
    检查指定链的相邻残基 CA 原子是否距离
    """
    ca_atoms = [res['CA'] for res in chain if 'CA' in res]
    
    dist_arr = []
    for i in range(len(ca_atoms) - 1):
        dist = np.linalg.norm(ca_atoms[i].coord - ca_atoms[i + 1].coord)
        dist_arr.append(dist)
    return np.array(dist_arr)

def get_peptide_valid(chain, threshold=3.8):
    """
    检查指定链的相邻残基 CA 原子是否距离均 < threshold。
    """
    ca_atoms = [res['CA'] for res in chain if 'CA' in res]
    
    for i in range(len(ca_atoms) - 1):
        dist = np.linalg.norm(ca_atoms[i].coord - ca_atoms[i + 1].coord)
        if dist > threshold:
            # print(f"Invalid CA-CA distance between residue {i} and {i+1}: {dist:.2f} Å")
            return False
    return True

def diff_ratio(str1, str2):
    # str1, str2 = get_seq(chain1), get_seq(chain2)
    # Create a SequenceMatcher object
    seq_matcher = difflib.SequenceMatcher(None, str1, str2)

    # Calculate the difference ratio
    return seq_matcher.ratio()

def get_novel(chain1, chain2, tm_thr=0.5, seq_thr=0.5):
    """
    判断单条预测多肽是否 Novel:
      - TM-score ≤ tm_thr
      - 且 序列相似度(identity) ≤ seq_thr
    """
    if chain1 is None or chain2 is None:
        return False

    # 1) 结构相似度：TM-score
    tm = get_tm(chain1, chain2)
    if tm is None or tm > tm_thr:
        return False

    # 2) 序列相似度
    seq_pred = get_seq(chain1)
    seq_ref = get_seq(chain2)
    sid = diff_ratio(seq_pred, seq_ref)
    if sid > seq_thr:
        return False

    return True

import itertools

def compute_diversity_avg(chains, seqs):
    """
    计算一组多肽的 diversity（平均版）：
      diversity = (1 / M) * Σ_{i<j} [(1 - TM_score[i,j]) * (1 - seq_identity[i,j])]
    其中 M = C(N,2) 是总的对数。
    """
    assert len(chains) == len(seqs)
    n = len(chains)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i, j in itertools.combinations(range(n), 2):
        tm  = get_tm(chains[i], chains[j])
        sid = diff_ratio(seqs[i], seqs[j])
        if tm is None:
            continue
        total += (1.0 - tm) * (1.0 - sid)
        count += 1

    return total / count if count > 0 else 0.0

#######################################

#RMSD and Tm

#######################################
def align_chains(chain1, chain2):
    reslist1 = []
    reslist2 = []
    for residue1,residue2 in zip(chain1.get_residues(),chain2.get_residues()):
        if is_aa(residue1) and residue1.has_id('CA'): # at least have CA
            reslist1.append(residue1)
            reslist2.append(residue2)
    return reslist1,reslist2

def get_rmsd(chain1, chain2):
    # chain1 = get_chain_from_pdb(pdb1, chain_id1)
    # chain2 = get_chain_from_pdb(pdb2, chain_id2)
    if chain1 is None or chain2 is None:
        return None
    super_imposer = Superimposer()
    # pos1 = np.array([atom.get_coord() for atom in chain1.get_atoms() if atom.name == 'CA'])
    # pos2 = np.array([atom.get_coord() for atom in chain2.get_atoms() if atom.name == 'CA'])
    # rmsd1 = np.sqrt(np.sum((pos1 - pos2)**2) / len(pos1))
    super_imposer.set_atoms([atom for atom in chain1.get_atoms() if atom.name == 'CA'],
                            [atom for atom in chain2.get_atoms() if atom.name == 'CA'])
    rmsd2 = super_imposer.rms
    return rmsd2

def get_tm(chain1,chain2):
    # chain1 = get_chain_from_pdb(pdb1, chain_id1)
    # chain2 = get_chain_from_pdb(pdb2, chain_id2)
    pos1 = np.array([atom.get_coord() for atom in chain1.get_atoms() if atom.name == 'CA'])
    pos2 = np.array([atom.get_coord() for atom in chain2.get_atoms() if atom.name == 'CA'])
    tm_results = tmtools.tm_align(pos1, pos2, 'A'*len(pos1), 'A'*len(pos2))
    # print(dir(tm_results))
    return tm_results.tm_norm_chain2

def get_traj_chain(pdb, chain):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    chain2id = {chain.id:i for i,chain in enumerate(structure)}
    traj = md.load(pdb)
    chain_indices = traj.topology.select(f"chainid {chain2id[chain]}")
    traj = traj.atom_slice(chain_indices)
    return traj

def get_psi_chi(pdb, chain):
    traj_chain = get_traj_chain(pdb, chain)
    _, phi_angles = md.compute_phi(traj_chain)
    _, psi_angles = md.compute_psi(traj_chain)
    return psi_angles, phi_angles

def get_second_stru(pdb,chain):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    chain2id = {chain.id:i for i,chain in enumerate(structure)}
    traj = md.load(pdb)
    chain_indices = traj.topology.select(f"chainid {chain2id[chain]}")
    traj = traj.atom_slice(chain_indices)
    return md.compute_dssp(traj,simplified=True)

def get_ss(pdb1,chain_id1,pdb2,chain_id2):
    traj1,traj2 = get_traj_chain(pdb1,chain_id1),get_traj_chain(pdb2,chain_id2)
    ss1,ss2 = md.compute_dssp(traj1,simplified=True),md.compute_dssp(traj2,simplified=True)
    return (ss1==ss2).mean()

def get_bind_site(pdb,chain_id):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    peps = [atom for res in structure[chain_id] for atom in res if atom.get_name() == 'CA']
    recs = [atom for chain in structure if chain.get_id()!=chain_id for res in chain for atom in res if atom.get_name() == 'CA']
    search = NeighborSearch(recs)
    near_res = []
    for atom in peps:
        near_res += search.search(atom.get_coord(), 10.0, level='R')
    near_res = set([res.get_id()[1] for res in near_res])
    return near_res

def get_bind_ratio(pdb1, pdb2, chain_id1, chain_id2):
    near_res1,near_res2 = get_bind_site(pdb1,chain_id1),get_bind_site(pdb2,chain_id2)
    return len(near_res1.intersection(near_res2))/(len(near_res2)+1e-10) # last one is gt

def get_seq(chain):
    return seq1("".join([residue.get_resname() for residue in chain])) # ignore is_aa,used for extract seq from genrated pdb

def get_mpnn_seqs(path):
    fastas = []
    for record in SeqIO.parse(path, "fasta"):
        tmp = [c for c in str(record.seq)]
        fastas.append(tmp)
    return fastas

