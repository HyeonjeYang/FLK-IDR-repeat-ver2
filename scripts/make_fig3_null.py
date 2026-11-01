#!/usr/bin/env python3
"""Figure 3: composition-preserving permutation nulls (SECONDARY analysis).
Shows the global null (all residues shuffled) and the segmental/local null
(shuffled only within fixed 25-aa blocks) for the primary window (1-150),
the segmental null's block-phase-offset robustness check, and a compact
text summary of the two sensitivity windows (1-107, 1-182).
Answers: how much of the observed FLK repeat multiplicity can be explained
by global composition versus more local composition, and is that answer an
artifact of where the local blocks happen to start?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat.architecture import find_motif_occurrences
from flk_repeat.null_model import block_bounds
from flk_repeat.paths import RESULTS_DIR, FIGURES_DIR
from flk_repeat.sequence import load_flk_record

PRIMARY_WINDOW = "primary_1_150"
PHASE_ROBUSTNESS_P_THRESHOLD = 0.01
SCHEMATIC_SEED = 7  # illustration only; not used by any statistic
PHASE_OFFSETS = (0, 5, 10, 15, 20)

# Residue classes for the schematic (colour-blind-safe; Q and E are the
# dominant residues of the window and get the two strongest hues).
RESIDUE_CLASSES = [
    ("Q", set("Q"), "#2166ac"),
    ("D/E (acidic)", set("DE"), "#d6604d"),
    ("Y/F/W (aromatic)", set("YFW"), "#fdb863"),
    ("L/V/I/M/A (hydrophobic)", set("LVIMA"), "#4d4d4d"),
    ("P", set("P"), "#92c5de"),
    ("other", None, "#e0e0e0"),
]


def _residue_colour(aa):
    for _, members, colour in RESIDUE_CLASSES:
        if members is None or aa in members:
            return colour


def _draw_residue_row(ax, seq, y, label, boundaries=None, repeat_starts=None):
    for i, aa in enumerate(seq):
        ax.add_patch(Rectangle((i, y), 1, 0.8, facecolor=_residue_colour(aa), edgecolor="none"))
    for s in repeat_starts or []:
        ax.add_patch(Rectangle((s - 1, y - 0.08), 8, 0.96, fill=False, edgecolor="black", lw=1.4))
    for b in boundaries or []:
        ax.plot([b, b], [y - 0.12, y + 0.92], color="black", lw=1.2)
    ax.text(-2, y + 0.4, label, ha="right", va="center", fontsize=9)


def _schematic_panel(ax, window_seq):
    rng = np.random.default_rng(SCHEMATIC_SEED)
    n = len(window_seq)
    seg0 = block_bounds(n, 25, 0)
    seg_shuffled = list(window_seq)
    for lo, hi in seg0:
        seg_shuffled[lo:hi] = rng.permutation(seg_shuffled[lo:hi])
    rows = [
        (window_seq, "Observed FLK 1-150\n(repeats boxed)", None, find_motif_occurrences(window_seq)),
        ("".join(rng.permutation(list(window_seq))), "Global null:\nall 150 residues shuffled", None, None),
        ("".join(seg_shuffled), "Segmental null: shuffled only\nwithin 25-residue blocks",
         [lo for lo, _ in seg0[1:]], None),
    ]
    y = 5.4
    for seq, label, bounds, reps in rows:
        _draw_residue_row(ax, seq, y, label, bounds, reps)
        y -= 1.25
    # block-phase robustness: where the block boundaries fall at each offset
    for k, off in enumerate(PHASE_OFFSETS):
        yy = 1.35 - k * 0.32
        ax.plot([0, n], [yy, yy], color="#bbbbbb", lw=0.8)
        for lo, _ in block_bounds(n, 25, off)[1:]:
            ax.plot([lo, lo], [yy - 0.12, yy + 0.12], color="black", lw=1.2)
        ax.text(n + 1.5, yy, f"offset {off}", fontsize=7.5, va="center")
    ax.text(-2, 0.7, "Block-phase robustness:\nsegmental null repeated with\nthe block grid shifted", ha="right",
            va="center", fontsize=9)
    for s in find_motif_occurrences(window_seq):
        ax.axvspan(s - 1, s + 7, ymin=0.0, ymax=0.25, color="#fdb863", alpha=0.25, lw=0)
    ax.set_xlim(-1, n + 12)
    ax.set_ylim(-0.1, 6.5)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax.set_xlabel("Boundary after residue (FLK 1–150; 0 = before residue 1)")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.legend(handles=[Patch(color=c, label=name) for name, _, c in RESIDUE_CLASSES],
              ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.set_title("A  What the null models shuffle", loc="left", fontweight="bold", fontsize=10.5, pad=22)


def _hist_panel(ax, hist_row, summary_row, color, title):
    """The observed-T8 marker carries its value as a legend entry, placed
    at upper-center -- the empty gap between the short middle bar (T8=2)
    and the marker line at the right edge, and clear of the tall T8=1 bar
    on the left -- so the legend never overlaps a bar or the red line."""
    values = hist_row.T8_value.to_numpy()
    counts = hist_row.n_permutations.to_numpy()
    ax.bar(values, counts, width=0.8, color=color, edgecolor="white")
    for v, c in zip(values, counts):
        # nudge the label off the bar that coincides with the observed-T8
        # marker line, so text and line never sit on top of each other
        ha = "right" if v == summary_row.observed_T8 else "center"
        x_off = -0.15 if v == summary_row.observed_T8 else 0
        ax.text(v + x_off, c * 1.35, f"{c:,}", ha=ha, va="bottom", fontsize=7.5, color="#333333")
    ci_str = f"{summary_row.empirical_p_ci_lower:.2g}, {summary_row.empirical_p_ci_upper:.2g}"
    ax.axvline(summary_row.observed_T8, color="#d73027", lw=2, ls="--",
               label=f"observed $T_8$={int(summary_row.observed_T8)}, "
                     f"$p_{{+}}={summary_row.p_emp_T8_display}$\n"
                     f"95% CI for tail $P$ [{ci_str}]")
    ax.set_yscale("log")
    ax.set_ylim(top=counts.max() * 6)  # headroom so the bar-count labels never clip
    ax.set_xlabel("$T_8$ = max exact-8-mer multiplicity")
    ax.set_ylabel("Permutations (log scale)")
    ax.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9, loc="upper center")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)


def main():
    summary = pd.read_csv(RESULTS_DIR / "flk_null_model_summary.tsv", sep="\t")
    hist = pd.read_csv(RESULTS_DIR / "flk_null_model_histogram.tsv", sep="\t")
    summary["p_emp_T8_display"] = summary["empirical_p"].map(lambda p: f"{p:.2g}")

    global_row = summary[(summary.window == PRIMARY_WINDOW) & (summary.null_model == "global")].iloc[0]
    segmental_rows = summary[(summary.window == PRIMARY_WINDOW) &
                              (summary.null_model == "segmental")].sort_values("phase_offset")
    seg_row0 = segmental_rows[segmental_rows.phase_offset == 0].iloc[0]
    global_hist = hist[(hist.window == PRIMARY_WINDOW) & (hist.null_model == "global")]
    seg_hist0 = hist[(hist.window == PRIMARY_WINDOW) & (hist.null_model == "segmental") &
                      (hist.phase_offset == 0)]
    phase_robust = bool((segmental_rows.empirical_p < PHASE_ROBUSTNESS_P_THRESHOLD).all())

    fig = plt.figure(figsize=(13.2, 10.7), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.78, 1, 0.73])
    axS = fig.add_subplot(gs[0, :])
    axA, axB = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axC, axD = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    _schematic_panel(axS, load_flk_record()["sequence"][:150])
    _hist_panel(axA, global_hist, global_row, "#4393c3", "B  Global composition null")
    _hist_panel(axB, seg_hist0, seg_row0, "#66a61e", "C  Segmental composition null")

    axC.plot(segmental_rows.phase_offset, segmental_rows.empirical_p, "o-", color="#66a61e")
    axC.axhline(PHASE_ROBUSTNESS_P_THRESHOLD, color="black", lw=1, ls="--")
    axC.set_yscale("log")
    axC.set_ylim(top=PHASE_ROBUSTNESS_P_THRESHOLD * 3)
    axC.text(segmental_rows.phase_offset.max(), PHASE_ROBUSTNESS_P_THRESHOLD * 1.4,
              f"robustness threshold ({PHASE_ROBUSTNESS_P_THRESHOLD})",
              color="black", fontsize=8, ha="right", va="bottom")
    axC.set_xticks(segmental_rows.phase_offset)
    axC.set_xlabel("block-grid phase offset (aa)")
    axC.set_ylabel("Plus-one $p_+$ for $T_8$, segmental null")
    verdict = "all offsets below 0.01" if phase_robust else "offset threshold crossed"
    axC.text(0.02, 0.90, verdict, fontsize=8, fontweight="bold",
              color=("#1a9850" if phase_robust else "#d73027"), transform=axC.transAxes)
    axC.set_title("D  Block-phase robustness", loc="left", fontweight="bold", fontsize=10.5)

    axD.axis("off")
    header = f"{'window (aa)':<15}{'model':<11}{'offset':>7}{'p+':>11}{'tail P 95% CI':>21}"
    lines = [header, "-" * len(header)]
    window_labels = {"primary_1_150": "1–150", "sensitivity_1_107": "1–107",
                     "sensitivity_1_182": "1–182"}
    for _, row in summary.sort_values(["window", "null_model", "phase_offset"]).iterrows():
        offset_str = "" if pd.isna(row.phase_offset) else str(int(row.phase_offset))
        ci_str = f"[{row.empirical_p_ci_lower:.2g}, {row.empirical_p_ci_upper:.2g}]"
        lines.append(f"{window_labels[row.window]:<15}{row.null_model:<11}{offset_str:>7}{row.empirical_p:>11.2g}{ci_str:>21}")
    axD.text(0, 0.98, "\n".join(lines), family="monospace", fontsize=7.2,
             va="top", transform=axD.transAxes)
    axD.set_title("E  Null-model summary", loc="left", fontweight="bold", fontsize=10.5)

    fig.suptitle("Composition-preserving null models for FLK repeat multiplicity",
                 fontweight="bold", fontsize=13)

    fig.savefig(FIGURES_DIR / "fig3_composition_null.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'fig3_composition_null.png'}")


if __name__ == "__main__":
    main()
