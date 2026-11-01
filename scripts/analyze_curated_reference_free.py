#!/usr/bin/env python3
"""Reference-free 8-mer survey of the existing 22 curated candidates.

This is a descriptive bridge to the expanded sampling requested in revision.
It uses only cached sequences, each candidate's own pre-KH1 region, and no
alignment or A. thaliana motif for the best-family search. The separate
LEPQQYEV count is explicitly reference-dependent. No orthology or
statistical-significance inference is made from this table.
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.homologs import parse_raw_candidates  # noqa: E402
from flk_repeat.paths import RAW_HOMOLOG_CANDIDATES_FASTA, HOMOLOG_PROVENANCE_TSV, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from flk_repeat.provenance import sha256_of_sequence  # noqa: E402
from flk_repeat.repeats import best_repeat_family, motif_family_copies  # noqa: E402
from flk_repeat.figure_labels import scientific_name  # noqa: E402

OUT = RESULTS_DIR / "expanded" / "curated_reference_free_repeats.tsv"
FIG = FIGURES_DIR / "figS2_reference_free_curated.png"


def analyze() -> pd.DataFrame:
    seqs = parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    provenance = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    rows = []
    for r in provenance.itertuples(index=False):
        if not r.included:
            continue
        seq = seqs[r.accession]
        assert len(seq) == r.length and sha256_of_sequence(seq) == r.sha256
        kh1 = int(r.kh1_start_1based)
        assert 8 < kh1 <= len(seq)
        region = seq[:kh1 - 1]
        kmers = Counter(region[i:i + 8] for i in range(len(region) - 7))
        exact = sorted(kmers.items(), key=lambda x: (-x[1], x[0]))[0]
        one = best_repeat_family(region, 8, 1)
        two = best_repeat_family(region, 8, 2)
        motif = motif_family_copies(region, "LEPQQYEV", 3)
        rows.append({
            "species": r.species, "accession": r.accession,
            "orthology_status": r.orthology_status,
            "pre_KH1_length": len(region),
            "T8_exact_overlapping": exact[1], "best_exact_8mer": exact[0],
            "best_8mer_1_mismatch_seed": one["seed"],
            "A8_1_overlapping": one["n_overlapping_matches"],
            "best_8mer_1_mismatch_nonoverlapping_copies": one["n_copies_nonoverlapping"],
            "best_8mer_1_mismatch_copies": ";".join(one["copies"]),
            "best_8mer_1_mismatch_starts": ";".join(map(str, one["starts_1based"])),
            "best_8mer_2_mismatch_seed": two["seed"],
            "A8_2_overlapping": two["n_overlapping_matches"],
            "best_8mer_2_mismatch_nonoverlapping_copies": two["n_copies_nonoverlapping"],
            "LEPQQYEV_family_3_mismatch_nonoverlapping_copies": len(motif),
            "LEPQQYEV_family_3_mismatch_windows": ";".join(f"{pos}:{win}:{mm}" for pos, win, mm in motif),
        })
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame) -> None:
    cols = ["T8_exact_overlapping", "best_8mer_1_mismatch_nonoverlapping_copies",
            "best_8mer_2_mismatch_nonoverlapping_copies",
            "LEPQQYEV_family_3_mismatch_nonoverlapping_copies"]
    titles = ["Exact 8-mer\nT8 (overlap)", "Best family\n≤1 mismatch\n(non-overlap)",
              "Best family\n≤2 mismatches\n(non-overlap)",
              "LEPQQYEV family\n≤3 mismatches\n(non-overlap)"]
    values = df[cols].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(11.2, 9.0), constrained_layout=True)
    ax.imshow(values, cmap="Blues", vmin=0, vmax=max(5, values.max()), aspect="auto")
    ax.set_xticks(range(4), titles)
    ax.set_yticks(range(len(df)), [scientific_name(s) for s in df.species], fontsize=8.2)
    ax.tick_params(axis="x", labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    for i in range(len(df)):
        for j in range(4):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] >= 5 else "#242424")
        ax.text(3.65, i, df.iloc[i]["best_8mer_1_mismatch_seed"],
                ha="left", va="center", fontsize=8, family="monospace")
    ax.text(3.65, -1.25, "Best-family seed\n(<=1 mismatch)", ha="left", va="bottom", fontsize=9)
    ax.set_xlim(-0.55, 5.35)
    ax.set_ylim(len(df) - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="#e7e7e7", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("8-mer counts in the curated FLK candidate set", fontsize=13,
                 fontweight="bold")
    fig.savefig(FIG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> pd.DataFrame:
    df = analyze()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    make_figure(df)
    print(f"wrote {OUT} ({len(df)} candidates) and {FIG}")
    return df


if __name__ == "__main__":
    main()
