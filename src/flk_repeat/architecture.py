"""FLK sequence architecture: exact LEPQQYEV occurrences, spacing, local
amino-acid composition, and validated KH-domain / disorder annotations.

N-terminal analysis windows (chosen a priori, never selected by which
boundary gives the strongest statistical result -- see docs/METHODS.md):

- primary_1_150: Mockler et al. (2004) describe FLK residues 1-150 as
  glutamine-rich and report the three LEPQQYEV repeats within it. This is
  the literature-defined primary window.
- sensitivity_1_107: UniProt's own annotated disordered region 1 (see
  data/provenance/flk_sequence_provenance.json).
- sensitivity_1_182: the first IUPred2A "long" disorder segment (score
  >= 0.5 for >= 20 consecutive residues), independently recomputed here
  from the audited IUPred2A cache and confirmed to span residues 1-182.

Only primary_1_150 is used for the headline statistical result; the other
two are robustness/sensitivity checks reported side by side, never used to
pick the "best" boundary.
"""
import re

MOTIF = "LEPQQYEV"

WINDOWS = {
    "primary_1_150": (1, 150),
    "sensitivity_1_107": (1, 107),
    "sensitivity_1_182": (1, 182),
}


def find_motif_occurrences(seq: str, motif: str = MOTIF) -> list[int]:
    """1-based start coordinates of every exact, non-overlapping-by-construction
    (the motif does not self-overlap) occurrence of `motif` in `seq`."""
    return [m.start() + 1 for m in re.finditer(re.escape(motif), seq)]


def repeat_spacing(starts: list[int], motif_length: int = len(MOTIF)) -> list[int]:
    """Gap, in residues, between the end of one repeat and the start of the
    next (0 = immediately adjacent)."""
    return [starts[i + 1] - (starts[i] + motif_length) for i in range(len(starts) - 1)]


def window_slice(seq: str, start_1based: int, end_1based: int) -> str:
    """Inclusive 1-based slice, e.g. window_slice(seq, 1, 150) == seq[:150]."""
    return seq[start_1based - 1:end_1based]


def window_composition(seq: str, start_1based: int, end_1based: int) -> dict:
    """E/Q/Y composition of an inclusive 1-based residue window."""
    sub = window_slice(seq, start_1based, end_1based)
    n = len(sub)
    e, q, y = sub.count("E"), sub.count("Q"), sub.count("Y")
    return {
        "start_1based": start_1based,
        "end_1based": end_1based,
        "length": n,
        "E_count": e,
        "Q_count": q,
        "Y_count": y,
        "E_fraction": e / n,
        "Q_fraction": q / n,
        "Y_fraction": y / n,
        "EQY_fraction": (e + q + y) / n,
    }


def near_motif_windows(seq: str, motif: str = MOTIF, max_mismatches: int = 1) -> list[tuple]:
    """Post-hoc, motif-specific: 1-based start positions of every overlapping
    8-aa window with at most `max_mismatches` substitutions relative to
    `motif` (Hamming distance on equal-length strings; no indels). Distinct
    from the general max-repeat statistic T_8 used in the null model."""
    k = len(motif)
    hits = []
    for i in range(len(seq) - k + 1):
        window = seq[i:i + k]
        mismatches = sum(1 for a, b in zip(window, motif) if a != b)
        if mismatches <= max_mismatches:
            hits.append((i + 1, window, mismatches))
    return hits
