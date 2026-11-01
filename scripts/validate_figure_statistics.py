#!/usr/bin/env python3
"""Recalculate Figure 3 tail statistics from cached permutation histograms.

The Monte Carlo point estimate uses the plus-one correction. The exact
Clopper-Pearson interval applies to the underlying exceedance probability
estimated by k/N, so the two quantities are labeled separately in figures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.null_model import clopper_pearson_ci  # noqa: E402
from flk_repeat.paths import RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "expanded" / "figure3_statistical_audit.tsv"


def main() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS_DIR / "flk_null_model_summary.tsv", sep="\t")
    hist = pd.read_csv(RESULTS_DIR / "flk_null_model_histogram.tsv", sep="\t")
    rows = []
    for s in summary.itertuples(index=False):
        h = hist[(hist.window == s.window) & (hist.null_model == s.null_model)]
        h = h[h.phase_offset.isna()] if pd.isna(s.phase_offset) else h[h.phase_offset == s.phase_offset]
        n = int(h.n_permutations.sum())
        k = int(h.loc[h.T8_value >= s.observed_T8, "n_permutations"].sum())
        p_plus_one = (k + 1) / (n + 1)
        ci_lo, ci_hi = clopper_pearson_ci(k, n)
        mean_t8 = float(np.average(h.T8_value, weights=h.n_permutations))
        checks = {
            "n": n == int(s.N_permutations),
            "p": np.isclose(p_plus_one, s.empirical_p, rtol=0, atol=1e-15),
            "ci_lower": np.isclose(ci_lo, s.empirical_p_ci_lower, rtol=0, atol=1e-15),
            "ci_upper": np.isclose(ci_hi, s.empirical_p_ci_upper, rtol=0, atol=1e-15),
            "mean": np.isclose(mean_t8, s.null_mean_T8, rtol=0, atol=1e-12),
            "max": int(h.T8_value.max()) == int(s.null_max_T8),
        }
        rows.append({"window": s.window, "null_model": s.null_model,
                     "phase_offset": s.phase_offset, "N": n, "k_at_least_observed": k,
                     "observed_T8": int(s.observed_T8), "p_plus_one": p_plus_one,
                     "tail_probability_CI95_lower": ci_lo,
                     "tail_probability_CI95_upper": ci_hi,
                     "all_checks_pass": all(checks.values())})
        if not all(checks.values()):
            raise AssertionError(f"{s.window}/{s.null_model}/{s.phase_offset}: {checks}")
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"validated {len(out)} Figure 3 null distributions; wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
