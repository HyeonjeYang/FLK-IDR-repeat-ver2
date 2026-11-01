#!/usr/bin/env python3
"""Pruned display of the 102-tip KH-domain gene-family tree.

All tree inference and bootstrap calculations use the full focused set;
pruning here changes only what is displayed.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from Bio import Phylo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.figure_labels import scientific_name  # noqa: E402
from flk_repeat.paths import FIGURES_DIR, RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "expanded"
CONTROLS = {
    "Ananas comosus", "Brachypodium distachyon",
    "Beta vulgaris subsp. vulgaris", "Brassica oleracea var. oleracea",
    "Populus trichocarpa", "Solanum lycopersicum",
    "Tarenaya hassleriana", "Zea mays",
}
REFS = {"Q9SR13": "FLK", "Q9SZH4": "PEP", "Q9LZ82": "BTR1",
        "F4KDN0": "HEN4", "Q8W4B1": "RCF3"}


def fit_tip_labels(ax, fig, padding_px: float = 16) -> None:
    """Reserve enough x-axis space for the longest terminal label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_width = ax.get_window_extent(renderer).width
    left, right = ax.get_xlim()
    for label in ax.texts:
        text_width = label.get_window_extent(renderer).width
        available = axis_width - text_width - padding_px
        if available <= 0:
            raise ValueError("Tree label is wider than the plotting area")
        anchor = label.get_position()[0]
        right = max(right, left + (anchor - left) * axis_width / available)
    ax.set_xlim(left, right)


def main() -> None:
    audit = pd.read_csv(OUT / "expanded_phylogeny_audit.tsv", sep="\t")
    failed = set(audit.loc[audit.is_initial_selected_gene &
                           ~audit.passes_repeat_region_qc, "species"])
    keep = set(audit.loc[audit.is_initial_selected_gene &
                         audit.species.isin(failed | CONTROLS), "tree_accession"])
    keep.update(REFS)
    labels = {r.tree_accession: (scientific_name(r.species) + f"  {r.tree_accession}")
              for r in audit.itertuples(index=False) if r.tree_accession in keep}
    labels.update({a: scientific_name("Arabidopsis thaliana") + f"  {name} ({a})"
                   for a, name in REFS.items()})
    colors = {}
    for r in audit.itertuples(index=False):
        if r.tree_accession in keep:
            colors[labels[r.tree_accession]] = "#c14e35" if r.species in failed else "#3275a8"
    colors[labels["Q9SR13"]] = "#1d8f62"
    for a in REFS:
        if a != "Q9SR13":
            colors[labels[a]] = "#555555"

    tree = Phylo.read(OUT / "expanded_focus_kh_tree.nwk", "newick")
    for terminal in list(tree.get_terminals()):
        if terminal.name not in keep:
            tree.prune(terminal)
    tree.ladderize()
    fig, ax = plt.subplots(figsize=(12, 11.5))
    Phylo.draw(tree, label_func=lambda c: labels.get(c.name, "") if c.is_terminal() else "",
               label_colors=lambda label: colors.get(label, "black"),
               do_show=False, show_confidence=False, axes=ax)
    support = float(audit.focus_FLK_side_bootstrap_pct.iloc[0])
    ax.set_title(f"KH-domain tree: 14 selected genes with repeat-region QC failures\n"
                 f"FLK-side split: {support:.1f}% bootstrap support (400 replicates, 102-tip tree)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Corrected amino-acid distance across three FLK KH domains")
    ax.set_ylabel("")
    ax.set_yticks([])
    legend = [
        Line2D([0], [0], color="#c14e35", lw=3, label="Initial gene: repeat-region QC fails"),
        Line2D([0], [0], color="#3275a8", lw=3, label="QC-passing lineage control"),
        Line2D([0], [0], color="#1d8f62", lw=3,
               label=scientific_name("Arabidopsis thaliana") + " FLK"),
        Line2D([0], [0], color="#555555", lw=3,
               label=scientific_name("Arabidopsis thaliana") + " KH paralog"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8, frameon=False)
    fit_tip_labels(ax, fig)
    fig.text(0.02, 0.01,
             "Display pruned from the 102-tip tree. Red indicates repeat-region QC failure, "
             "not phylogenetic exclusion.\n"
             "A fuller isoform of the same " + scientific_name("Amborella trichopoda") +
             " gene was used for tree placement.",
             fontsize=8)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "figS5_expanded_phylogeny_qc.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path} ({len(tree.get_terminals())} displayed tips)")


if __name__ == "__main__":
    main()
