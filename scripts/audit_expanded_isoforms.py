#!/usr/bin/env python3
"""Audit all cached RefSeq isoforms of each initially selected FLK gene."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.expanded import athaliana_reference, choose_representatives, locate_kh1, parse_genpept  # noqa: E402
from flk_repeat.features import STANDARD_AA  # noqa: E402
from flk_repeat.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "expanded" / "selected_gene_isoform_qc.tsv"


def main() -> pd.DataFrame:
    seq_dir = DATA_DIR / "expanded" / "sequences"
    records = parse_genpept((seq_dir / "candidate_hits.gp").read_text(encoding="utf-8"))
    records.update(parse_genpept((seq_dir / "representative_gene_isoforms.gp").read_text(encoding="utf-8")))
    selected = choose_representatives()
    reference, kh1_span = athaliana_reference()
    rows = []
    for rep in selected:
        for acc in rep["isoform_accessions"].split(";"):
            record = records[acc]
            seq = record["sequence"]
            kh1 = locate_kh1(seq, reference, kh1_span)
            pre = kh1["kh1_start_1based"] - 1
            issues = []
            if not seq.startswith("M"):
                issues.append("missing_start_M")
            if not set(seq) <= set(STANDARD_AA):
                issues.append("nonstandard_residue")
            if not 350 <= len(seq) <= 750:
                issues.append("length_outside_350_750")
            if kh1["kh1_fraction_aligned"] < 0.8:
                issues.append("KH1_coverage_below_0.8")
            if pre < 75:
                issues.append("pre_KH1_below_75_aa")
            rows.append({"species": rep["species"], "gene_id": rep["gene_id"],
                         "accession": acc, "is_initial_representative": acc == rep["accession"],
                         "protein_length": len(seq), "pre_KH1_length": pre,
                         "KH1_fraction_aligned": kh1["kh1_fraction_aligned"],
                         "passes_repeat_region_qc": not issues, "qc_issues": ";".join(issues)})
    out = pd.DataFrame(rows).sort_values(["species", "is_initial_representative", "pre_KH1_length"],
                                         ascending=[True, False, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    failed = set(out.loc[out.is_initial_representative & ~out.passes_repeat_region_qc, "species"])
    rescue = out[out.species.isin(failed) & out.passes_repeat_region_qc]
    print(f"{len(out)} isoforms of 40 initially selected genes; "
          f"{len(failed)} initial isoforms fail QC; "
          f"{rescue.species.nunique()} have a QC-passing isoform of the same gene")
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
