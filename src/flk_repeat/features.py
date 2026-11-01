r"""Physicochemical (sequence-property) descriptors of an intrinsically
disordered region, used to ask whether the FLK N-terminal IDR keeps its
chemistry even where its exact sequence has diverged (revision, Reviewer 1
comment 1 and the author's own follow-up request).

Two kinds of descriptor are kept deliberately separate, because the
composition-preserving null models in `null_model.py` treat them
differently:

- **Composition descriptors** depend only on which residues are present
  (hydropathy, hydrophilicity, acidic/basic/charged/polar/aromatic
  fractions, net charge, isoelectric point, compositional entropy). Any
  shuffle of a sequence leaves them unchanged.
- **Order (patterning) descriptors** also depend on where residues sit
  (sequence charge decoration SCD, residue-class patterning z-scores,
  maximal exact 8-mer multiplicity T_8). A shuffle changes them.

Scales and definitions
----------------------
- Kyte-Doolittle hydropathy (Kyte & Doolittle 1982, J Mol Biol 157:105):
  mean over residues; higher = more hydrophobic.
- Hopp-Woods hydrophilicity (Hopp & Woods 1981, PNAS 78:3824): mean over
  residues; higher = more hydrophilic. Reported alongside Kyte-Doolittle
  because the two scales were derived independently and weight residues
  differently (e.g. Q/N are strongly hydrophilic on KD but nearly neutral on
  Hopp-Woods).
- Charged residues: D, E (-1); K, R (+1). Histidine is treated as neutral
  at pH 7, as in Das & Pappu (2013).
- FCR = fraction of charged residues; NCPR = net charge per residue
  = (f+ - f-).
- Polar (uncharged hydrophilic) residues: S, T, N, Q, C. Aromatic: F, W, Y.
  Hydrophobic: A, I, L, M, F, V, W.
- Isoelectric point: Biopython `ProteinAnalysis.isoelectric_point`
  (Bjellqvist pK set).
- Residue-class patterning z-score (after the NARDINI approach, Cohan et
  al. 2022, J Mol Biol 434:167373): for a residue class X, the local
  clustering statistic is $\delta_X = \overline{(b_i - f_X)^2}$, where
  $b_i$ is the fraction of X in the i-th overlapping blob and $f_X$ the
  fraction in the whole region, averaged over blob sizes 5 and 6. It is
  reported as a z-score against composition-preserving shuffles of the
  same region: z > 0 = X residues more clustered than chance, z < 0 =
  more evenly spaced than chance. This replaces the Das & Pappu (2013)
  kappa parameter, whose normalization is unstable for regions with very
  few basic residues (the FLK N-terminus has almost none; kappa exceeded 1
  for 4% of A. thaliana disordered regions in a check of this code).
- SCD (Sawle & Ghosh 2015, J Chem Phys 143:085101):
  $\mathrm{SCD} = \frac{1}{N}\sum_{i<j} q_i q_j |i-j|^{1/2}$; more negative
  = more blocky segregation of opposite charges.
- Compositional Shannon entropy (bits) over the 20 standard residues.
"""
import math
from collections import Counter

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
HOPP_WOODS = {
    "A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0,
    "G": 0.0, "H": -0.5, "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5,
    "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5,
}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}
POLAR = set("STNQC")
AROMATIC = set("FWY")
HYDROPHOBIC = set("AILMFVW")

PATTERNING_CLASSES = {
    "acidic": set("DE"),
    "polar": POLAR,
    "glutamine": set("Q"),
    "aromatic": AROMATIC,
    "proline": set("P"),
    "hydrophobic": HYDROPHOBIC,
}
PATTERNING_BLOBS = (5, 6)
PATTERNING_SHUFFLES = 1000
PATTERNING_SEED = 20261020

COMPOSITION_FEATURES = [
    "hydropathy_kd", "hydrophilicity_hw", "fraction_hydrophobic", "fraction_polar",
    "fraction_acidic", "fraction_basic", "fcr", "ncpr", "isoelectric_point",
    "fraction_aromatic", "fraction_Q", "fraction_E", "fraction_Y", "fraction_P",
    "fraction_G", "fraction_S", "composition_entropy_bits",
]
ORDER_FEATURES = ["scd"] + [f"patterning_z_{name}" for name in PATTERNING_CLASSES] + [
    "t8_max_exact_8mer_multiplicity"]
ALL_FEATURES = ["length"] + COMPOSITION_FEATURES + ORDER_FEATURES


def clean(seq: str) -> str:
    """Uppercase, standard residues only (gaps/X/stops removed)."""
    return "".join(c for c in seq.upper() if c in STANDARD_AA)


def _fraction(seq: str, residues) -> float:
    return sum(1 for c in seq if c in residues) / len(seq)


def composition_entropy(seq: str) -> float:
    counts = Counter(seq)
    n = len(seq)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _blob_clustering(indicator: np.ndarray, blobs=PATTERNING_BLOBS) -> np.ndarray:
    """delta_X for every row of a (B, L) 0/1 indicator matrix, averaged
    over blob sizes."""
    f = indicator.mean(axis=1, keepdims=True)
    csum = np.concatenate([np.zeros((indicator.shape[0], 1)), np.cumsum(indicator, axis=1)], axis=1)
    values = []
    for g in blobs:
        blob_frac = (csum[:, g:] - csum[:, :-g]) / g
        values.append(((blob_frac - f) ** 2).mean(axis=1))
    return np.mean(values, axis=0)


def patterning_z(seq: str, residues, n_shuffles: int = PATTERNING_SHUFFLES,
                 seed: int = PATTERNING_SEED) -> float:
    """Clustering of a residue class relative to composition-preserving
    shuffles (see module docstring). NaN when the class is absent, fills
    the whole region, or the region is shorter than the largest blob."""
    seq = clean(seq)
    x = np.array([1.0 if c in residues else 0.0 for c in seq])
    if len(x) < max(PATTERNING_BLOBS) or x.sum() in (0, len(x)):
        return float("nan")
    observed = _blob_clustering(x[None, :])[0]
    rng = np.random.default_rng(seed)
    null = _blob_clustering(rng.permuted(np.tile(x, (n_shuffles, 1)), axis=1))
    sd = null.std(ddof=1)
    return float("nan") if sd == 0 else float((observed - null.mean()) / sd)


def scd(seq: str) -> float:
    seq = clean(seq)
    q = [CHARGE.get(c, 0) for c in seq]
    idx = [i for i, x in enumerate(q) if x != 0]
    total = 0.0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            total += q[i] * q[j] * math.sqrt(j - i)
    return total / len(seq)


def max_exact_kmer_multiplicity(seq: str, k: int = 8) -> int:
    counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    return max(counts.values()) if counts else 0


def compute_features(seq: str, include_patterning: bool = True) -> dict:
    """All descriptors for one region (see module docstring). Patterning
    z-scores need shuffles; `include_patterning=False` skips them for
    large background sets."""
    seq = clean(seq)
    if not seq:
        raise ValueError("empty sequence after removing non-standard residues")
    n = len(seq)
    counts = Counter(seq)
    fp = sum(counts[c] for c in "KR") / n
    fm = sum(counts[c] for c in "DE") / n
    patterning = {f"patterning_z_{name}": (patterning_z(seq, res) if include_patterning else float("nan"))
                  for name, res in PATTERNING_CLASSES.items()}
    return {
        "length": n,
        "hydropathy_kd": sum(KYTE_DOOLITTLE[c] for c in seq) / n,
        "hydrophilicity_hw": sum(HOPP_WOODS[c] for c in seq) / n,
        "fraction_hydrophobic": _fraction(seq, HYDROPHOBIC),
        "fraction_polar": _fraction(seq, POLAR),
        "fraction_acidic": fm,
        "fraction_basic": fp,
        "fcr": fp + fm,
        "ncpr": fp - fm,
        "isoelectric_point": ProteinAnalysis(seq).isoelectric_point(),
        "fraction_aromatic": _fraction(seq, AROMATIC),
        "fraction_Q": counts["Q"] / n,
        "fraction_E": counts["E"] / n,
        "fraction_Y": counts["Y"] / n,
        "fraction_P": counts["P"] / n,
        "fraction_G": counts["G"] / n,
        "fraction_S": counts["S"] / n,
        "composition_entropy_bits": composition_entropy(seq),
        "scd": scd(seq),
        **patterning,
        "t8_max_exact_8mer_multiplicity": max_exact_kmer_multiplicity(seq, 8),
    }
