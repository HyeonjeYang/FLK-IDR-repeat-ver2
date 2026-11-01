#!/usr/bin/env python3
"""Descriptive pre-KH1 chemistry across the cached 22-candidate set.

The A. thaliana IDR background gives context, not a length-matched null or
an evolutionary conservation test. All plotted features depend only on
composition, so shuffling the same region cannot change them.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat import features as ft  # noqa: E402
from flk_repeat.homologs import parse_raw_candidates  # noqa: E402
from flk_repeat.paths import RAW_HOMOLOG_CANDIDATES_FASTA, HOMOLOG_PROVENANCE_TSV, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from flk_repeat.provenance import sha256_of_sequence  # noqa: E402
from flk_repeat.figure_labels import scientific_name  # noqa: E402

OUT = RESULTS_DIR / "expanded" / "curated_pre_kh1_chemistry.tsv"
BACKGROUND = RESULTS_DIR / "expanded" / "athaliana_idr_background_features.tsv"
FIG = FIGURES_DIR / "figS3_curated_chemistry.png"


def analyze() -> pd.DataFrame:
    seqs = parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    provenance = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    rows = []
    for r in provenance.itertuples(index=False):
        if not r.included:
            continue
        seq = seqs[r.accession]
        assert len(seq) == r.length and sha256_of_sequence(seq) == r.sha256
        region = seq[:int(r.kh1_start_1based) - 1]
        desc = ft.compute_features(region, include_patterning=False)
        rows.append({"species": r.species, "accession": r.accession,
                     "orthology_status": r.orthology_status,
                     **{k: desc[k] for k in ["length", *ft.COMPOSITION_FEATURES]}})
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, bg: pd.DataFrame) -> None:
    panels = [
        ("fraction_Q", "Glutamine fraction", "Fraction Q"),
        ("hydropathy_kd", "Mean hydropathy", "Kyte-Doolittle scale"),
        ("ncpr", "Net charge per residue", "NCPR"),
        ("composition_entropy_bits", "Composition entropy (bits)", "Bits"),
    ]
    colors = {"REFERENCE": "#d95f02", "HIGH_CONFIDENCE_ORTHOLOG": "#1b9e77",
              "SUPPORTED_ORTHOLOG_CANDIDATE": "#7570b3", "UNCERTAIN": "#555555"}
    legend_labels = {"REFERENCE": "Reference", "HIGH_CONFIDENCE_ORTHOLOG": "High-confidence candidate",
                     "SUPPORTED_ORTHOLOG_CANDIDATE": "Supported candidate", "UNCERTAIN": "Uncertain candidate"}
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 9.2), sharey=True, constrained_layout=True)
    y = list(range(len(df)))
    for ax, (field, title, xlabel) in zip(axes, panels):
        q5, median, q95 = bg[field].quantile([0.05, 0.5, 0.95])
        ax.axvspan(q5, q95, color="#e4e4e4", zorder=0)
        ax.axvline(median, color="#777777", lw=1, ls="--", zorder=1)
        for status, grp in df.groupby("orthology_status"):
            ax.scatter(grp[field], grp.index, s=24, color=colors[status],
                       label=legend_labels[status], zorder=2)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(axis="x", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(df) - 0.5, -0.5)
    axes[0].set_yticks(y, [scientific_name(s) for s in df.species], fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Composition of the curated FLK candidates' pre-KH1 regions", fontsize=13,
                 fontweight="bold")
    fig.savefig(FIG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> pd.DataFrame:
    df = analyze()
    bg = pd.read_csv(BACKGROUND, sep="\t")
    assert len(bg) == 13253
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    make_figure(df, bg)
    print(f"wrote {OUT} ({len(df)} candidates) and {FIG}")
    return df


if __name__ == "__main__":
    main()
