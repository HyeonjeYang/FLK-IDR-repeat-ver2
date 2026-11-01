#!/usr/bin/env python3
"""Figure 2: reference-mapped similarity of FLK N-terminal windows among
curated homologs/ortholog candidates (PRIMARY analysis). Species are
ordered by curation group (close Arabidopsis relatives -> other
Brassicaceae -> distant eudicot candidates -> rice), the order used
throughout data/provenance/homolog_provenance.tsv -- an evolutionary-
distance-informed ordering, not a formally inferred phylogeny.

The displayed repeat windows and identity/similarity values are the
PRIMARY KH1-anchored pairwise mapping (results/
flk_homolog_repeat_comparison.tsv), unchanged. Cells for which an
independent full-length Clustal Omega MSA (results/
flk_alignment_validation.tsv) yielded a materially different repeat
assignment are additionally outlined and marked -- an uncertainty
annotation, not a relabeling: the underlying pairwise value shown is never
altered or replaced by the MSA-derived one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat.paths import RESULTS_DIR, FIGURES_DIR, HOMOLOG_PROVENANCE_TSV
from flk_repeat.figure_labels import scientific_name

STATUS_ABBREV = {
    "REFERENCE": "REF",
    "HIGH_CONFIDENCE_ORTHOLOG": "HIGH_CAND",
    "SUPPORTED_ORTHOLOG_CANDIDATE": "SUPP_CAND",
    "UNCERTAIN": "UNCERTAIN",
}
SENSITIVE_EDGE_COLOR = "#404040"


def _outline_cell(ax, row, col, **kwargs):
    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                            edgecolor=SENSITIVE_EDGE_COLOR, linewidth=1.8, zorder=5, **kwargs))


def main():
    df = pd.read_csv(RESULTS_DIR / "flk_homolog_repeat_comparison.tsv", sep="\t")
    provenance = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    validation = pd.read_csv(RESULTS_DIR / "flk_alignment_validation.tsv", sep="\t")

    species_order = [s for s in provenance.loc[provenance.included, "species"] if s != "Arabidopsis thaliana"]
    status_by_species = df.drop_duplicates("species").set_index("species").orthology_status.to_dict()

    n_sp = len(species_order)
    identity = np.full((n_sp, 3), np.nan)
    blosum = np.full((n_sp, 3), np.nan)
    labels = np.empty((n_sp, 3), dtype=object)
    sensitive = np.zeros((n_sp, 3), dtype=bool)
    for i, sp in enumerate(species_order):
        for r in range(3):
            row = df[(df.species == sp) & (df.repeat_index == r + 1)]
            if len(row):
                aligned, ref = row.aligned_sequence.iloc[0], row.reference_sequence.iloc[0]
                identity[i, r] = sum(1 for a, b in zip(aligned, ref) if a == b) / len(ref)
                blosum[i, r] = row.blosum62_similarity.iloc[0]
                labels[i, r] = row.aligned_sequence.iloc[0]
            else:
                labels[i, r] = ""
            vrow = validation[(validation.species == sp) & (validation.repeat_index == r + 1)]
            if len(vrow):
                sensitive[i, r] = vrow.alignment_validation_status.iloc[0] == "MATERIAL_DISAGREEMENT"
    n_sensitive_cells = int(sensitive.sum())

    fig, (axL, axI, axB) = plt.subplots(
        1, 3, figsize=(13.5, 0.34 * n_sp + 2.5),
        gridspec_kw={"width_ratios": [2.1, 1, 1]}, constrained_layout=True)

    # Left: species (italic, own text object) / status / aligned-sequence
    # text panel. Species names are drawn separately from the monospace
    # status+repeat block so they can be italicized (proportional-font
    # italic text cannot share a single monospace-aligned string).
    axL.axis("off")
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    SPECIES_X, DATA_X = 0.0, 0.40
    row_h = 1.0 / (n_sp + 1.3)
    header_y = 1.0 - 0.4 * row_h
    axL.text(SPECIES_X, header_y, "species", family="monospace", fontsize=8.2,
              fontweight="bold", transform=axL.transAxes)
    axL.text(DATA_X, header_y, f"{'tier':<12}{'ref 1':>10}{'ref 2':>10}{'ref 3':>10}",
              family="monospace", fontsize=8.2, fontweight="bold", transform=axL.transAxes)
    for i, sp in enumerate(species_order):
        y = header_y - (i + 1) * row_h
        status = STATUS_ABBREV.get(status_by_species[sp], status_by_species[sp][:10])
        cells = [labels[i, r].replace("-", ".") + ("*" if sensitive[i, r] else "") for r in range(3)]
        axL.text(SPECIES_X, y, scientific_name(sp), fontsize=8.2, transform=axL.transAxes)
        axL.text(DATA_X, y, f"{status:<12}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}",
                  family="monospace", fontsize=8, transform=axL.transAxes)
    axL.set_title("A  Reference-mapped windows", loc="left", fontweight="bold", fontsize=10)
    axL.text(0, header_y - (n_sp + 0.9) * row_h,
              "* / outlined cell = alignment-sensitive (see caption)",
              fontsize=6.8, style="italic", color=SENSITIVE_EDGE_COLOR, transform=axL.transAxes, va="top")

    cmap = plt.get_cmap("RdYlBu_r")
    imI = axI.imshow(identity, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axI.set_yticks(range(n_sp)); axI.set_yticklabels([])
    axI.set_xticks([0, 1, 2]); axI.set_xticklabels(["ref 1", "ref 2", "ref 3"], fontsize=8)
    axI.set_title("B  Exact identity", loc="left", fontweight="bold", fontsize=10)
    fig.colorbar(imI, ax=axI, fraction=0.06, pad=0.03, label="identity fraction")

    imB = axB.imshow(blosum, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axB.set_yticks(range(n_sp)); axB.set_yticklabels([])
    axB.set_xticks([0, 1, 2]); axB.set_xticklabels(["ref 1", "ref 2", "ref 3"], fontsize=8)
    axB.set_title("C  Normalized BLOSUM62 score", loc="left", fontweight="bold", fontsize=10)
    fig.colorbar(imB, ax=axB, fraction=0.06, pad=0.03, label="score / reference self-score")

    for i in range(n_sp):
        for r in range(3):
            if sensitive[i, r]:
                _outline_cell(axI, i, r)
                _outline_cell(axB, i, r)
    # Legend for the outline marker lives in the panel-A note below (no
    # in-axes legend here, to avoid ever overlapping a heatmap cell).

    fig.suptitle("Reference-mapped window similarity in FLK candidates", fontweight="bold", fontsize=13)

    fig.savefig(FIGURES_DIR / "fig2_repeat_conservation.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'fig2_repeat_conservation.png'} ({n_sensitive_cells} alignment-sensitive cells marked)")


if __name__ == "__main__":
    main()
