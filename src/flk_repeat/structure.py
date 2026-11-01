"""AlphaFold model of A. thaliana FLK (AF-Q9SR13-F1, model v6), used only
as a positive control for residue-covariation analysis: if MIp couplings
in the FLK alignment recover residue contacts in the structured KH
domains, the alignment carries enough coevolutionary signal for a
negative result in the disordered N-terminus to be informative.

Contacts: C-beta (C-alpha for glycine) distance < 8 A, sequence separation
>= 6, both residues with pLDDT >= 70 (pLDDT is stored in the B-factor
column of AlphaFold PDB files).
"""
import numpy as np

from .paths import DATA_DIR

AF_MODEL = DATA_DIR / "expanded" / "structure" / "AF-Q9SR13-F1-model_v6.pdb"
CONTACT_CUTOFF = 8.0
MIN_SEPARATION = 6
MIN_PLDDT = 70.0


def read_cb_and_plddt(path=AF_MODEL) -> tuple[np.ndarray, np.ndarray]:
    """(coords[n, 3], plddt[n]) indexed by residue number - 1."""
    coords, plddt = {}, {}
    for line in open(path, encoding="utf-8"):
        if not line.startswith("ATOM"):
            continue
        atom, resname, resnum = line[12:16].strip(), line[17:20], int(line[22:26])
        if atom == "CB" or (atom == "CA" and resname == "GLY"):
            coords[resnum] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            plddt[resnum] = float(line[60:66])
    n = max(coords)
    xyz = np.full((n, 3), np.nan)
    conf = np.full(n, np.nan)
    for r, c in coords.items():
        xyz[r - 1] = c
        conf[r - 1] = plddt[r]
    return xyz, conf


def contact_map(path=AF_MODEL) -> tuple[np.ndarray, np.ndarray]:
    """Boolean contact matrix and per-residue 'confidently structured' mask."""
    xyz, plddt = read_cb_and_plddt(path)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    confident = plddt >= MIN_PLDDT
    idx = np.arange(len(xyz))
    sep = np.abs(idx[:, None] - idx[None, :])
    contacts = (d < CONTACT_CUTOFF) & (sep >= MIN_SEPARATION) & confident[:, None] & confident[None, :]
    return contacts, confident
