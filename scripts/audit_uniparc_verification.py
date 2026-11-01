#!/usr/bin/env python3
"""Audit UniParc exact-sequence cross-references for 40 RefSeq representatives."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "expanded" / "uniparc_sequence_verification.tsv"


def main() -> pd.DataFrame:
    reciprocal = pd.read_csv(RESULTS_DIR / "expanded" / "reciprocal_blast_audit.tsv", sep="\t")
    source = json.loads((DATA_DIR / "expanded" / "verification" /
                         "uniparc_crossrefs.json").read_text(encoding="utf-8"))
    rows = []
    for r in reciprocal.itertuples(index=False):
        u = source[r.query_accession]
        base = r.query_accession.split(".")[0]
        matched = [x for x in u["cross_references"]
                   if x["database"] == "RefSeq" and x["id"].split(".")[0] == base]
        rows.append({
            "species": r.species, "accession": r.query_accession,
            "sequence_md5": u["md5"], "uniparc_id": u["upi"],
            "has_exact_sequence_uniparc_entry": bool(u["upi"]),
            "has_matching_refseq_crossref": bool(matched),
            "has_active_matching_refseq_crossref": any(x["active"] for x in matched),
            "matching_refseq_taxon_ids": ";".join(sorted(
                {str(x["taxon_id"]) for x in matched if x["taxon_id"]})),
            "interpretation": (
                "EXACT_SEQUENCE_ACTIVE_REFSEQ" if any(x["active"] for x in matched)
                else "EXACT_SEQUENCE_REFSEQ_INACTIVE" if matched
                else "EXACT_SEQUENCE_IN_UNIPARC_OTHER_DB" if u["upi"]
                else "NO_UNIPARC_CHECKSUM_MATCH"
            ),
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    print(out.interpretation.value_counts().to_string())
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
