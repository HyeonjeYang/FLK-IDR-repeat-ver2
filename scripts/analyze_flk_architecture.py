#!/usr/bin/env python3
"""FLK sequence architecture: validate the audited Q9SR13 record, locate the
three exact LEPQQYEV repeats, and report N-terminal composition and the
validated KH-domain / UniProt-disorder annotations.

N-terminal composition across the three predefined analysis windows
(primary 1-150, sensitivity 1-107, sensitivity 1-182) is written by
scripts/run_composition_null.py (results/flk_window_composition.tsv), next
to the null-model statistics that consume it.

Writes:
    results/flk_sequence_provenance.tsv
    results/flk_repeat_coordinates.tsv
    results/flk_domain_annotation.tsv
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import sequence as seqmod
from flk_repeat import architecture as arch
from flk_repeat.paths import RESULTS_DIR


def main():
    rec = seqmod.load_flk_record()
    prov = rec["provenance"]
    seq = rec["sequence"]

    # --- provenance table ---
    prov_df = pd.DataFrame([{
        "accession": rec["accession"],
        "protein_name": prov["protein_name"],
        "organism": prov["organism"],
        "gene": prov["gene"],
        "locus": prov["locus"],
        "length_aa": rec["length"],
        "reviewed": rec["reviewed"],
        "source": prov["source"],
        "retrieval_audit_date": prov["retrieval_audit_date"],
        "sequence_sha256_uppercase": rec["sha256"],
    }])
    prov_df.to_csv(RESULTS_DIR / "flk_sequence_provenance.tsv", sep="\t", index=False)

    # --- repeat coordinates ---
    starts = arch.find_motif_occurrences(seq)
    assert len(starts) == 3, f"expected exactly 3 LEPQQYEV copies, found {len(starts)}"
    gaps = arch.repeat_spacing(starts)
    rows = []
    for i, start in enumerate(starts, start=1):
        end = start + len(arch.MOTIF) - 1
        gap_after = gaps[i - 1] if i - 1 < len(gaps) else None
        rows.append({
            "repeat_index": i,
            "start_1based": start,
            "end_1based": end,
            "sequence": seq[start - 1:end],
            "gap_to_next_repeat_aa": gap_after,
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "flk_repeat_coordinates.tsv", sep="\t", index=False)

    # --- validated domain / disorder annotation (for Fig 1) ---
    dom_rows = []
    for d in prov["kh_domains_1based_inclusive"]:
        dom_rows.append({"feature_type": "KH_domain", "name": d["name"], "start": d["start"], "end": d["end"]})
    for i, d in enumerate(prov["uniprot_disordered_regions_1based_inclusive"], start=1):
        dom_rows.append({"feature_type": "UniProt_disordered_region", "name": f"disordered_{i}",
                          "start": d["start"], "end": d["end"]})
    pd.DataFrame(dom_rows).to_csv(RESULTS_DIR / "flk_domain_annotation.tsv", sep="\t", index=False)

    print(f"FLK repeats at (1-based): {starts}; spacing (aa between repeats): {gaps}")
    kh1_start = prov["kh_domains_1based_inclusive"][0]["start"]
    return {"sequence": seq, "starts": starts, "record": rec, "kh1_start_1based": kh1_start}


if __name__ == "__main__":
    main()
