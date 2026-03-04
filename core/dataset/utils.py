from core.openfold.utils import rigid_utils as ru
from core.dataset import residue_constants, protein
import string

Rigid = ru.Rigid
Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

to_numpy = lambda x: x.detach().cpu().numpy()
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])

def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)
