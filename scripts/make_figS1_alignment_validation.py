#!/usr/bin/env python3
"""Figure S1 (supplementary/validation, not a primary scientific figure):
agreement between the primary KH1-anchored pairwise repeat mapping and an
independent full-length Clustal Omega MSA (data/validation/
flk_full_length_clustalo.fasta). See docs/METHODS.md, "Independent
full-length MSA sensitivity check".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import homologs as hom
from flk_repeat.paths import RESULTS_DIR, FIGURES_DIR, HOMOLOG_PROVENANCE_TSV, DATA_DIR
from flk_repeat.figure_labels import scientific_name

MSA_PATH = DATA_DIR / "validation" / "flk_full_length_clustalo.fasta"
STATUS_CODE = {"EXACT_AGREEMENT": 0, "NONMATERIAL_DIFFERENCE": 1, "MATERIAL_DISAGREEMENT": 2}
# Colorblind-safe palette (avoids red-green confusion): blue / yellow / red
# instead of the original green / yellow / red -- blue vs. red stays fully
# distinguishable under deuteranopia and protanopia. Same 3-category format,
# same legend, same layout as before.
STATUS_COLOR = ["#2166ac", "#fee08b", "#d73027"]
STATUS_LABEL = ["exact agreement", "nonmaterial difference", "material disagreement\n(alignment-sensitive)"]

# One representative species per curation tier for the alignment-excerpt panel.
EXCERPT_ACCESSIONS = ["Q9SR13", "D7L277", "A0ACE2D236", "A0A087H6X3", "R0G4A9", "A0A6P5S226"]


def main():
    detail = pd.read_csv(RESULTS_DIR / "flk_alignment_validation.tsv", sep="\t")
    provenance = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    species_order = [s for s in provenance.loc[provenance.included, "species"] if s != "Arabidopsis thaliana"]

    n_sp = len(species_order)
    grid = np.full((n_sp, 3), np.nan)
    for i, sp in enumerate(species_order):
        for r in range(3):
            row = detail[(detail.species == sp) & (detail.repeat_index == r + 1)]
            if len(row):
                grid[i, r] = STATUS_CODE[row.alignment_validation_status.iloc[0]]

    fig = plt.figure(figsize=(13, 0.30 * n_sp + 4.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.30 * n_sp + 1.2, 2.6])
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    cmap = ListedColormap(STATUS_COLOR)
    axA.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    axA.set_yticks(range(n_sp))
    axA.set_yticklabels([scientific_name(s) for s in species_order], fontsize=8)
    axA.set_xticks([0, 1, 2])
    axA.set_xticklabels(["repeat 1", "repeat 2", "repeat 3"], fontsize=9)
    axA.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(STATUS_COLOR, STATUS_LABEL)],
               loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    axA.set_title("A  Pairwise–MSA agreement", loc="left", fontweight="bold", fontsize=10.5)

    # Panel B: raw alignment excerpt spanning all three repeat columns, for
    # a handful of representative species across the evolutionary-distance
    # gradient (not exhaustive -- see panel A / results/flk_alignment_
    # validation.tsv for every species).
    msa = hom.parse_fasta(MSA_PATH)
    athal_gapped = msa["Q9SR13"]
    col_of_pos = hom.ungapped_position_to_column(athal_gapped)
    lo, hi = col_of_pos[45], col_of_pos[86]
    acc_to_species = provenance.set_index("accession").species.to_dict()

    axB.axis("off")
    axB.set_title("B  Representative MSA excerpt", loc="left", fontweight="bold", fontsize=10.5)
    for i, acc in enumerate(EXCERPT_ACCESSIONS):
        label = "A. thaliana (reference)" if acc == "Q9SR13" else acc_to_species.get(acc, acc)
        y = 1.0 - (i + 0.7) / (len(EXCERPT_ACCESSIONS) + 0.4)
        axB.text(0.0, y, scientific_name(label), fontsize=8.5, transform=axB.transAxes)
        axB.text(0.34, y, msa[acc][lo:hi], family="monospace", fontsize=8.5, transform=axB.transAxes)

    fig.suptitle("Independent alignment sensitivity analysis", fontweight="bold", fontsize=13)

    fig.savefig(FIGURES_DIR / "figS1_alignment_validation.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'figS1_alignment_validation.png'}")


if __name__ == "__main__":
    main()
