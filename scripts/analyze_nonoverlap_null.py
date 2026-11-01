#!/usr/bin/env python3
"""Independent FLK shuffle check with a non-overlapping exact 8-mer statistic.

The vectorized overlapping statistic filters the very rare candidate draws;
only those draws require a non-overlapping count in Python. This preserves
the exact non-overlapping result while keeping 1e6 draws practical.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.fast_null import ALPHABET, _batched_permutations, batch_max_kmer_multiplicity, encode  # noqa: E402
from flk_repeat.homologs import parse_raw_candidates  # noqa: E402
from flk_repeat.null_model import block_bounds, clopper_pearson_ci  # noqa: E402
from flk_repeat.paths import RAW_HOMOLOG_CANDIDATES_FASTA  # noqa: E402
from scripts.analyze_revision_claims import exact_kmer_counts  # noqa: E402


def run(n: int, seed: int, block_size: int | None, batch: int = 20_000) -> dict:
    seq = parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)["Q9SR13"][:150]
    codes = encode(seq)
    bounds = block_bounds(len(codes), block_size, 0) if block_size else None
    rng = np.random.default_rng(seed)
    overlap = 0
    nonoverlap = 0
    seeds = Counter()
    for done in range(0, n, batch):
        b = min(batch, n - done)
        permutations = _batched_permutations(codes, rng, b, bounds)
        t8 = batch_max_kmer_multiplicity(permutations, 8)
        hits = np.flatnonzero(t8 >= 3)
        overlap += len(hits)
        for index in hits:
            perm_seq = "".join(ALPHABET[v] for v in permutations[index])
            _t, best, t8_no = exact_kmer_counts(perm_seq)
            seeds[best] += 1
            nonoverlap += t8_no >= 3
    assert exact_kmer_counts(seq)[2] == 3
    return {"null_model": "global" if block_size is None else "25aa_block_offset_0",
            "N": n, "seed": seed, "observed_T8_nonoverlap": 3,
            "overlap_exceedances": overlap, "nonoverlap_exceedances": nonoverlap,
            "nonoverlap_p_plus_one": (nonoverlap + 1) / (n + 1),
            "nonoverlap_CI95_lower": clopper_pearson_ci(nonoverlap, n)[0],
            "nonoverlap_CI95_upper": clopper_pearson_ci(nonoverlap, n)[1],
            "overlap_exceedance_best_seeds": ";".join(f"{k}:{v}" for k, v in sorted(seeds.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    rows = [run(args.n, args.seed, None), run(args.n, args.seed, 25)]
    out = ROOT / "results" / "expanded" / "nonoverlap_null_summary.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
