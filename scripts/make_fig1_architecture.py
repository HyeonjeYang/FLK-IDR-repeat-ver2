#!/usr/bin/env python3
"""Figure 1: FLK sequence architecture -- full-length disorder profile, KH
domains, the three LEPQQYEV repeat positions, and local E/Q/Y composition
in the N-terminal region.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import sequence as seqmod
from flk_repeat import architecture as arch
from flk_repeat.paths import RESULTS_DIR, FIGURES_DIR

WINDOW = 25  # sliding window for the local E/Q/Y composition track


def sliding_eqy_fraction(seq: str, window: int = WINDOW) -> np.ndarray:
    n = len(seq)
    frac = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        sub = seq[lo:hi]
        frac[i] = sum(sub.count(a) for a in "EQY") / len(sub)
    return frac


def main():
    rec = seqmod.load_flk_record()
    seq, n = rec["sequence"], rec["length"]
    idr = np.array(seqmod.load_iupred_scores(n))

    coords = pd.read_csv(RESULTS_DIR / "flk_repeat_coordinates.tsv", sep="\t")
    domains = pd.read_csv(RESULTS_DIR / "flk_domain_annotation.tsv", sep="\t")
    eqy = sliding_eqy_fraction(seq)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), constrained_layout=True,
                              gridspec_kw={"height_ratios": [2.2, 0.9, 1.3, 1.5]})

    # Panel A: IUPred2A disorder profile
    axA = axes[0]
    x = np.arange(1, n + 1)
    axA.plot(x, idr, color="#2166ac", lw=1.1)
    axA.axhline(0.5, color="grey", lw=0.8, ls="--")
    axA.fill_between(x, 0, idr, where=idr >= 0.5, color="#2166ac", alpha=0.15, step=None)
    axA.set_ylim(0, 1)
    axA.set_xlim(1, n)
    axA.tick_params(labelbottom=False)
    axA.set_ylabel("IUPred2A score")
    axA.set_title("A  Predicted disorder and repeat positions", loc="left",
                   fontweight="bold", fontsize=10.5)

    # Panel B: domain / repeat map
    axB = axes[1]
    axB.set_xlim(1, n)
    axB.set_ylim(0, 1)
    axB.tick_params(labelbottom=False)
    axB.set_yticks([])
    for _, d in domains[domains.feature_type == "UniProt_disordered_region"].iterrows():
        axB.add_patch(Rectangle((d.start, 0.55), d.end - d.start + 1, 0.35, color="#2166ac", alpha=0.35))
    for _, d in domains[domains.feature_type == "KH_domain"].iterrows():
        axB.add_patch(Rectangle((d.start, 0.05), d.end - d.start + 1, 0.4, color="#d73027"))
        axB.text((d.start + d.end) / 2, 0.25, d["name"], ha="center", va="center", fontsize=7, color="white")
    for _, r in coords.iterrows():
        axB.add_patch(Rectangle((r.start_1based, 0.0), r.end_1based - r.start_1based + 1, 1.0,
                                 color="#f46d43", zorder=5))
    axB.text(1, -0.35, "orange = LEPQQYEV repeats   red = KH domain   blue = UniProt-annotated disordered region",
              fontsize=7.5, style="italic", transform=axB.transData, clip_on=False)
    axB.set_title("B  Feature map", loc="left", fontweight="bold", fontsize=10.5)

    # Panel C: local E/Q/Y composition track (N-terminal region emphasized)
    axC = axes[2]
    axC.plot(x, eqy, color="#66a61e", lw=1.0)
    axC.axvspan(1, arch.WINDOWS["primary_1_150"][1], color="grey", alpha=0.10)
    for name, ls in [("sensitivity_1_107", ":"), ("sensitivity_1_182", ":")]:
        boundary = arch.WINDOWS[name][1]
        axC.axvline(boundary, color="grey", lw=0.9, ls=ls)
        axC.text(boundary, 1.02, str(boundary), fontsize=6.5, color="grey", ha="center", clip_on=False)
    axC.set_ylim(0, 1)
    axC.set_xlim(1, n)
    axC.set_ylabel(f"local E+Q+Y frac.\n({WINDOW}-aa window)")
    axC.set_xlabel("Residue (1-based)")
    axC.set_title("C  Local E+Q+Y composition", loc="left", fontweight="bold", fontsize=10.5)
    axC.text(1, -0.28, "grey band = primary window (1-150)   dotted lines = sensitivity windows (1-107, 1-182)",
              fontsize=7, style="italic", transform=axC.transData, clip_on=False)

    # Panel D: annotation-derived exitron model, shown on long-protein coordinates.
    axD = axes[3]
    axD.set_title("D  Annotated in-frame exitron in the repeat array", loc="left",
                  fontweight="bold", fontsize=10.5)
    axD.set_xlim(40, 94)
    axD.set_ylim(0, 1.15)
    axD.set_yticks([0.34, 0.82], ["553-aa model", "577-aa model"])
    axD.tick_params(axis="y", length=0, labelsize=8)
    axD.set_xticks([51, 58, 63, 70, 75, 82], ["51", "58", "63", "70", "75", "82"])
    axD.set_xlabel("Residue in 577-aa reference (1-based)")
    axD.plot([45, 89], [0.82, 0.82], color="#444444", lw=5, solid_capstyle="butt")
    for i, start in enumerate((51, 63, 75), 1):
        axD.add_patch(Rectangle((start - 0.5, 0.69), 8, 0.26,
                                facecolor="#f46d43", edgecolor="#9e3d1b", lw=0.8))
        axD.text(start + 3.5, 0.82, str(i), ha="center", va="center", color="white",
                 fontsize=8, fontweight="bold")
    axD.plot([45, 57.5], [0.34, 0.34], color="#444444", lw=5, solid_capstyle="butt")
    axD.plot([81.5, 89], [0.34, 0.34], color="#444444", lw=5, solid_capstyle="butt")
    axD.plot([50.5, 57.5], [0.34, 0.34], color="#f46d43", lw=9, solid_capstyle="butt")
    axD.plot([81.5, 82.5], [0.34, 0.34], color="#f46d43", lw=9, solid_capstyle="butt")
    axD.add_patch(FancyArrowPatch((57.5, 0.34), (81.5, 0.34),
                                  connectionstyle="arc3,rad=-0.24", arrowstyle="-",
                                  color="#2166ac", lw=1.7))
    axD.text(69.5, 0.14, "72-nt GT–AG exitron; modeled junction 57|82",
             ha="center", va="center", color="#2166ac", fontsize=8)
    axD.text(69.5, 1.04, "LEPQQYEV × 3  →  junction-spanning LEPQQYEV × 1",
             ha="center", va="center", fontsize=8)
    axD.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(r"Sequence architecture of $\mathit{Arabidopsis\ thaliana}$ FLK",
                 fontweight="bold", fontsize=13)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig1_flk_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'fig1_flk_architecture.png'}")


if __name__ == "__main__":
    main()
