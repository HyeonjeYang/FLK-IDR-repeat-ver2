#!/usr/bin/env python3
"""SECONDARY analysis: composition-preserving permutation nulls for FLK
N-terminal sequence windows, plus the analytical motif-specific probability
(exact) and a Poisson-approximation sanity check. See
src/flk_repeat/null_model.py for full definitions and docs/METHODS.md for
why this is secondary to the evolutionary comparison in
analyze_repeat_conservation.py.

Three windows are analyzed (chosen a priori -- see
src/flk_repeat/architecture.py:WINDOWS): the literature-defined primary
window (1-150) and two sensitivity windows (1-107, 1-182). All three get a
global (fully-shuffled) permutation null; only the primary window also gets
a segmental (block-shuffled, block size 25) null, to ask whether the
repeat's multiplicity is still unusual once broad local composition is held
fixed.

Segmental-null block-phase robustness check: the repeats start at residues
51/63/75, so the default 25-aa block grid (0-25/25-50/.../125-150) places a
boundary inside or immediately next to the third repeat -- a possible
boundary artifact. The segmental null is therefore repeated at 5 predefined
block-phase offsets (0, 5, 10, 15, 20 aa, chosen before looking at any
result) rather than run once at the default phase; the offset is never
chosen to maximize significance.

Writes:
    results/flk_window_composition.tsv
    results/flk_null_model_summary.tsv
    results/flk_null_model_histogram.tsv
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import null_model as nm
from flk_repeat import architecture as arch
from flk_repeat.paths import FLK_SEQUENCE_JSON, RESULTS_DIR

PRIMARY_WINDOW = "primary_1_150"

# Fixed a priori, before inspecting any result -- never chosen to maximize
# significance. 0 is the default (unshifted) block grid.
SEGMENTAL_PHASE_OFFSETS = [0, 5, 10, 15, 20]

# A segmental-null result is treated as "robust to block phase" if every
# tested offset's empirical p-value stays under this threshold -- an order
# of magnitude looser than the global null's own p_emp_T8, not a threshold
# tuned to this particular dataset.
PHASE_ROBUSTNESS_P_THRESHOLD = 0.01


def main(n_permutations: int = nm.DEFAULT_N_PERMUTATIONS, seed: int = nm.DEFAULT_SEED,
         block_size: int = nm.DEFAULT_BLOCK_SIZE):
    flk_seq = json.loads(FLK_SEQUENCE_JSON.read_text(encoding="utf-8"))["sequence"]["value"]

    window_seqs = {name: arch.window_slice(flk_seq, start, end)
                   for name, (start, end) in arch.WINDOWS.items()}

    # --- window composition table (also records observed T8 per window) ---
    comp_rows = []
    for name, seq in window_seqs.items():
        comp = arch.window_composition(flk_seq, *arch.WINDOWS[name])
        t8_obs, t8_kmers = nm.max_kmer_multiplicity(seq)
        comp_rows.append({
            "window": name,
            "start_1based": comp["start_1based"],
            "end_1based": comp["end_1based"],
            "length": comp["length"],
            "E_count": comp["E_count"],
            "Q_count": comp["Q_count"],
            "Y_count": comp["Y_count"],
            "E_fraction": comp["E_fraction"],
            "Q_fraction": comp["Q_fraction"],
            "Y_fraction": comp["Y_fraction"],
            "EQY_fraction": comp["EQY_fraction"],
            "observed_T8": t8_obs,
            "observed_T8_kmers": ";".join(t8_kmers),
        })
    pd.DataFrame(comp_rows).to_csv(RESULTS_DIR / "flk_window_composition.tsv", sep="\t", index=False)

    # --- null models ---
    summary_rows = []
    hist_rows = []
    near_motif = arch.near_motif_windows(window_seqs[PRIMARY_WINDOW], max_mismatches=1)

    def add_run(window_name, seq, run_result, analytical):
        summary_rows.append({
            "window": window_name,
            "null_model": run_result["null_model"],
            "block_size": run_result["block_size"],
            "phase_offset": run_result["phase_offset"],
            "N_permutations": run_result["n_permutations"],
            "random_seed": run_result["seed"],
            "observed_T8": run_result["T8_observed"],
            "observed_T8_kmers": ";".join(run_result["T8_observed_kmers"]),
            "null_mean_T8": run_result["T8_null_mean"],
            "null_max_T8": run_result["T8_null_max"],
            "empirical_p": run_result["p_emp_T8"],
            "empirical_p_ci_lower": run_result["p_emp_T8_ci_lower"],
            "empirical_p_ci_upper": run_result["p_emp_T8_ci_upper"],
            "motif_exact_count_observed": run_result["motif_exact_count_observed"],
            "empirical_p_motif_count_ge_3": run_result["p_emp_motif_count_ge_3"],
            "analytical_P_window_equals_motif": analytical["P_window_equals_motif"],
            "analytical_E_X_expected_motif_windows": analytical["E_X_expected_motif_windows"],
            "poisson_approx_P_X_ge_3": analytical["poisson_approx_P_X_ge_3"],
        })
        for hb in nm.histogram_bins(run_result["t8_samples"]):
            hist_rows.append({"window": window_name, "null_model": run_result["null_model"],
                               "phase_offset": run_result["phase_offset"], **hb})

    for name, seq in window_seqs.items():
        analytical = nm.analytical_motif_probability(seq)
        global_run = nm.run_global_permutation_null(
            seq, n_permutations=n_permutations, seed=seed,
            progress=True, desc=f"global null [{name}]")
        add_run(name, seq, global_run, analytical)
        if name == PRIMARY_WINDOW:
            for offset in SEGMENTAL_PHASE_OFFSETS:
                seg_run = nm.run_segmental_permutation_null(
                    seq, block_size=block_size, n_permutations=n_permutations, seed=seed,
                    phase_offset=offset, progress=True,
                    desc=f"segmental null [{name}, offset={offset}]")
                add_run(name, seq, seg_run, analytical)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["n_windows_ge7of8_identity_to_motif_posthoc"] = summary_df["window"].apply(
        lambda w: len(near_motif) if w == PRIMARY_WINDOW else pd.NA)
    summary_df.to_csv(RESULTS_DIR / "flk_null_model_summary.tsv", sep="\t",
                      index=False, na_rep="NA")
    pd.DataFrame(hist_rows).to_csv(RESULTS_DIR / "flk_null_model_histogram.tsv", sep="\t", index=False)

    def fmt_ci(row):
        return f"p_emp_T8={row.empirical_p:.6g} (95% CI [{row.empirical_p_ci_lower:.3g}, {row.empirical_p_ci_upper:.3g}])"

    primary_global = summary_df[(summary_df.window == PRIMARY_WINDOW) & (summary_df.null_model == "global")].iloc[0]
    segmental_rows = summary_df[(summary_df.window == PRIMARY_WINDOW) & (summary_df.null_model == "segmental")]
    print(f"\n=== final null-model results (N={n_permutations:,} permutations, seed={seed}) ===")
    print(f"[{PRIMARY_WINDOW}] global null: T8_obs={primary_global.observed_T8}, {fmt_ci(primary_global)}")
    print(f"[{PRIMARY_WINDOW}] segmental null (block={block_size}), phase-offset sweep "
          f"{SEGMENTAL_PHASE_OFFSETS}:")
    for _, row in segmental_rows.sort_values("phase_offset").iterrows():
        print(f"    phase_offset={int(row.phase_offset):>2}  T8_obs={row.observed_T8}  {fmt_ci(row)}")

    phase_robust = bool((segmental_rows.empirical_p < PHASE_ROBUSTNESS_P_THRESHOLD).all())
    print(f"  segmental-null phase-offset robustness "
          f"(all p_emp_T8 < {PHASE_ROBUSTNESS_P_THRESHOLD}): {phase_robust}")
    if phase_robust:
        print("  -> segmental-null conclusion (T8=3 unusual under locally "
              "composition-preserving shuffling) is robust to block-grid origin.")
    else:
        print("  -> segmental-null result is phase-sensitive; downgrading it and "
              "retaining the global null as the primary null-model result (see docs/METHODS.md).")

    for _, row in summary_df[summary_df.null_model == "global"].iterrows():
        print(f"  sensitivity: window={row.window} global {fmt_ci(row)}")

    return {"summary": summary_df, "window_composition": pd.DataFrame(comp_rows), "phase_robust": phase_robust}


if __name__ == "__main__":
    main()
