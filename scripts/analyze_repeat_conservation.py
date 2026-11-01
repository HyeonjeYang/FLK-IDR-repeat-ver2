#!/usr/bin/env python3
"""Evolutionary conservation of the FLK LEPQQYEV repeat across a curated set
of homologs/ortholog candidates with UniProt provenance and KH-domain-
architecture QC (PRIMARY analysis -- see docs/METHODS.md). Not a claim of
formal orthology: see the orthology-confidence tiers in docs/METHODS.md.

Source data (checked into the repository, cached from UniProtKB REST API on
2026-08-26 -- see data/provenance/SOURCES.md for the curation methodology):
    data/raw/homolog_candidates_raw.fasta       (every candidate considered)
    data/provenance/homolog_provenance.tsv      (inclusion/exclusion + evidence)

This script only validates and re-derives from those two files; it performs
no network access.

Writes:
    data/curated/flk_homologs_curated.fasta
    results/flk_homolog_repeat_comparison.tsv
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import homologs as hom
from flk_repeat import architecture as arch
from flk_repeat.provenance import sha256_of_sequence
from flk_repeat.paths import (RAW_HOMOLOG_CANDIDATES_FASTA, CURATED_HOMOLOG_FASTA,
                               HOMOLOG_PROVENANCE_TSV, RESULTS_DIR)


def load_provenance_rows() -> list[dict]:
    df = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t", dtype={"kh1_start_1based": "Int64"})
    rows = df.to_dict(orient="records")
    for r in rows:
        r["included"] = bool(r["included"])
        r["kh1_start_1based"] = "" if pd.isna(r["kh1_start_1based"]) else int(r["kh1_start_1based"])
    return rows


def main(repeat_starts_1based, athaliana_kh1_start_1based):
    raw_seqs = hom.parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    provenance_rows = load_provenance_rows()

    # Cross-check the two source files against each other (SHA256 + length).
    for row in provenance_rows:
        acc = row["accession"]
        seq = raw_seqs[acc]
        assert len(seq) == row["length"], f"{acc}: length mismatch (fasta={len(seq)}, provenance={row['length']})"
        assert sha256_of_sequence(seq) == row["sha256"], f"{acc}: SHA256 mismatch"

    curated = hom.build_curated_fasta(raw_seqs, provenance_rows, CURATED_HOMOLOG_FASTA)

    comparison_rows = hom.repeat_comparison_table(raw_seqs, provenance_rows,
                                                   repeat_starts_1based, athaliana_kh1_start_1based)
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(RESULTS_DIR / "flk_homolog_repeat_comparison.tsv", sep="\t", index=False)

    n_included = sum(r["included"] for r in provenance_rows)
    n_excluded = len(provenance_rows) - n_included
    print(f"Homolog candidates: {n_included} included, {n_excluded} excluded "
          f"(see data/provenance/homolog_provenance.tsv)")
    print(f"Curated species: {len(curated)}")
    return {"raw_seqs": raw_seqs, "provenance_rows": provenance_rows, "curated": curated, "comparison": comp_df}


if __name__ == "__main__":
    import json
    from flk_repeat.paths import FLK_SEQUENCE_JSON, FLK_PROVENANCE_JSON
    flk_seq = json.loads(FLK_SEQUENCE_JSON.read_text(encoding="utf-8"))["sequence"]["value"]
    prov = json.loads(FLK_PROVENANCE_JSON.read_text(encoding="utf-8"))
    starts = arch.find_motif_occurrences(flk_seq)
    kh1_start = prov["kh_domains_1based_inclusive"][0]["start"]
    main(starts, kh1_start)
