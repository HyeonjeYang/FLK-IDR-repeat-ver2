#!/usr/bin/env python3
"""Audit reciprocal BLAST top hits for the expanded RefSeq candidates.

This is one orthology check, not a final orthology assignment. A top hit to
an A. thaliana FLK isoform establishes a reciprocal gene-level match; family
tree placement and sequence provenance are assessed separately.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.expanded import choose_representatives, parse_genpept  # noqa: E402
from flk_repeat.features import STANDARD_AA  # noqa: E402
from flk_repeat.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

FLK_GENE_ID = "819619"
VERIFY = DATA_DIR / "expanded" / "verification"
OUT = RESULTS_DIR / "expanded" / "reciprocal_blast_audit.tsv"


def _versioned_accession(description: dict) -> str:
    accession = description["accession"]
    fields = description.get("id", "").split("|")
    return fields[1] if len(fields) >= 3 and fields[1].startswith(accession + ".") else accession


def main() -> pd.DataFrame:
    reps = {r["accession"]: r for r in choose_representatives()}
    doc = json.loads((VERIFY / "reverse_blast_athaliana.json").read_text(encoding="utf-8"))
    subjects = parse_genpept((VERIFY / "reverse_blast_athaliana_subjects.gp").read_text(encoding="utf-8"))
    rows = []
    for result in doc["BlastOutput2"]:
        search = result["report"]["results"]["search"]
        query = search["query_title"]
        if query not in reps or not search.get("hits"):
            raise AssertionError(f"Missing representative or BLAST hits: {query}")
        top = search["hits"][0]
        desc = top["description"][0]
        accession = _versioned_accession(desc)
        subject = subjects.get(accession)
        if subject is None:
            raise AssertionError(f"Missing GenPept subject: {accession}")
        bit_score = max(h["bit_score"] for h in top["hsps"])
        rep = reps[query]
        issues = []
        if not rep["sequence"].startswith("M"):
            issues.append("missing_start_M")
        if not set(rep["sequence"]) <= set(STANDARD_AA):
            issues.append("nonstandard_residue")
        if not 350 <= rep["length"] <= 750:
            issues.append("length_outside_350_750")
        if rep["kh1_fraction_aligned"] < 0.8:
            issues.append("KH1_coverage_below_0.8")
        if rep["kh1_start_1based"] < 76:
            issues.append("pre_KH1_below_75_aa")
        rows.append({
            "species": rep["species"],
            "query_accession": query,
            "representative_length": rep["length"],
            "pre_KH1_length": rep["kh1_start_1based"] - 1,
            "KH1_fraction_aligned": rep["kh1_fraction_aligned"],
            "passes_repeat_region_qc": not issues,
            "repeat_region_qc_issues": ";".join(issues),
            "top_athaliana_accession": accession,
            "top_athaliana_gene": subject["gene"],
            "top_athaliana_gene_id": subject["gene_id"],
            "top_bit_score": bit_score,
            "top_evalue": min(h["evalue"] for h in top["hsps"]),
            "top_hit_is_flk_gene": subject["gene_id"] == FLK_GENE_ID,
            "n_reverse_hits": len(search["hits"]),
            "verification_status": "RECIPROCAL_TOP_HIT_FLK_ONLY",
        })
    if set(reps) != {r["query_accession"] for r in rows}:
        raise AssertionError("Reciprocal queries do not match selected representatives")
    out = pd.DataFrame(rows).sort_values("species")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"{int(out.top_hit_is_flk_gene.sum())}/{len(out)} reciprocal top hits map to FLK GeneID {FLK_GENE_ID}; "
          f"{int(out.passes_repeat_region_qc.sum())}/{len(out)} pass repeat-region QC; wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
