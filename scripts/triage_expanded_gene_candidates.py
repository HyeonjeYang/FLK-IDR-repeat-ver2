#!/usr/bin/env python3
"""Audit each BLAST-supported gene before expanded FLK tree inference.

One row per species and GeneID, with the longest pre-KH1 isoform available
in the cached GenPept records. This is sequence-architecture QC, not an
orthology assignment.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.expanded import athaliana_reference, locate_kh1, parse_genpept, choose_representatives  # noqa: E402
from flk_repeat.features import STANDARD_AA  # noqa: E402
from flk_repeat.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

SEQ = DATA_DIR / "expanded" / "sequences"
OUT = RESULTS_DIR / "expanded" / "gene_candidate_qc.tsv"


def main() -> pd.DataFrame:
    hits = pd.read_csv(DATA_DIR / "expanded" / "selection" / "candidate_hits.tsv", sep="\t")
    records = parse_genpept((SEQ / "candidate_hits.gp").read_text(encoding="utf-8"))
    records.update(parse_genpept((SEQ / "representative_gene_isoforms.gp").read_text(encoding="utf-8")))
    selected = {r["species"]: r for r in choose_representatives()}
    reference, kh1_span = athaliana_reference()
    rows = []
    for species, sp_hits in hits.groupby("sciname", sort=True):
        gene_groups: dict[str, list] = {}
        for h in sp_hits.itertuples(index=False):
            record = records.get(h.accession)
            if record is None:
                raise AssertionError(f"Missing GenPept: {h.accession}")
            gene_groups.setdefault(record["gene_id"] or h.accession, []).append(h)
        for gene_id, gene_hits in gene_groups.items():
            gene_accessions = {h.accession for h in gene_hits}
            if gene_id and not gene_id.startswith("XP_"):
                gene_accessions |= {acc for acc, r in records.items()
                                    if r["gene_id"] == gene_id}
            isoforms = []
            for acc in sorted(gene_accessions):
                rec = records[acc]
                kh1 = locate_kh1(rec["sequence"], reference, kh1_span)
                isoforms.append((kh1["kh1_start_1based"] - 1, len(rec["sequence"]), acc, kh1))
            pre_len, length, accession, kh1 = max(isoforms)
            _, tree_length, tree_accession, tree_kh1 = max(
                isoforms,
                key=lambda x: (x[3]["kh1_fraction_aligned"] >= 0.8,
                               x[1] >= 350, x[3]["kh1_fraction_aligned"],
                               x[1], x[0], x[2]))
            sequence = records[accession]["sequence"]
            issues = []
            if not sequence.startswith("M"):
                issues.append("missing_start_M")
            if not set(sequence) <= set(STANDARD_AA):
                issues.append("nonstandard_residue")
            if not 350 <= length <= 750:
                issues.append("length_outside_350_750")
            if kh1["kh1_fraction_aligned"] < 0.8:
                issues.append("KH1_coverage_below_0.8")
            if pre_len < 75:
                issues.append("pre_KH1_below_75_aa")
            top = max(gene_hits, key=lambda h: h.bit_score)
            rows.append({
                "species": species, "gene_id": gene_id,
                "gene_symbol": records[accession]["gene"],
                "representative_accession": accession,
                "tree_accession": tree_accession,
                "tree_protein_length": tree_length,
                "tree_KH1_fraction_aligned": tree_kh1["kh1_fraction_aligned"],
                "top_blast_accession": top.accession,
                "top_blast_bit_score": top.bit_score,
                "top_blast_evalue": top.evalue,
                "n_isoforms_considered": len(gene_accessions),
                "protein_length": length, "pre_KH1_length": pre_len,
                "KH1_fraction_aligned": kh1["kh1_fraction_aligned"],
                "passes_repeat_region_qc": not issues,
                "qc_issues": ";".join(issues),
                "is_initial_selected_gene": gene_id == selected[species]["gene_id"],
            })
    out = pd.DataFrame(rows).sort_values(["species", "top_blast_bit_score"],
                                         ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    failed_species = {s for s, r in selected.items()
                      if not any(x["is_initial_selected_gene"] and x["passes_repeat_region_qc"]
                                 for x in rows if x["species"] == s)}
    rescue = out[(out.species.isin(failed_species)) & out.passes_repeat_region_qc &
                 ~out.is_initial_selected_gene]
    print(f"{len(out)} gene-level candidates across {out.species.nunique()} species; "
          f"{len(failed_species)} initial representatives fail QC; "
          f"{rescue.species.nunique()} of those species have a QC-passing alternative")
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
