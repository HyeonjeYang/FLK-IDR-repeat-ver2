#!/usr/bin/env python3
"""One row per repeat-region QC failure with isoform/tree/crossref evidence."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.paths import RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "expanded"


def main() -> pd.DataFrame:
    reciprocal = pd.read_csv(OUT / "reciprocal_blast_audit.tsv", sep="\t")
    bad = reciprocal[~reciprocal.passes_repeat_region_qc].copy()
    genes = pd.read_csv(OUT / "gene_candidate_qc.tsv", sep="\t")
    isoforms = pd.read_csv(OUT / "selected_gene_isoform_qc.tsv", sep="\t")
    tree = pd.read_csv(OUT / "expanded_phylogeny_audit.tsv", sep="\t")
    sensitivity = pd.read_csv(OUT / "expanded_phylogeny_sensitivity.tsv", sep="\t")
    uniparc = pd.read_csv(OUT / "uniparc_sequence_verification.tsv", sep="\t")
    rows = []
    for r in bad.itertuples(index=False):
        g = genes[genes.species == r.species]
        alternatives = g[~g.is_initial_selected_gene & g.passes_repeat_region_qc]
        in_flk = tree[(tree.species == r.species) &
                      ~tree.is_initial_selected_gene &
                      tree.passes_repeat_region_qc &
                      tree.in_all_gene_FLK_side]
        initial = tree[(tree.species == r.species) &
                       tree.is_initial_selected_gene].iloc[0]
        sen = sensitivity[sensitivity.species == r.species].iloc[0]
        uni = uniparc[uniparc.species == r.species].iloc[0]
        same_gene_rescue = isoforms[(isoforms.species == r.species) &
                                    isoforms.passes_repeat_region_qc]
        tree_isoform = isoforms[(isoforms.species == r.species) &
                                (isoforms.accession == initial.tree_accession)].iloc[0]
        if len(in_flk):
            category = "FLK_SIDE_QC_ALTERNATIVE_UNRESOLVED_DUPLICATION"
        elif len(alternatives):
            category = "QC_ALTERNATIVES_OUTSIDE_FLK_SIDE"
        else:
            category = "NO_QC_PASSING_ALTERNATIVE"
        rows.append({
            "species": r.species,
            "repeat_representative_accession": r.query_accession,
            "tree_isoform_accession": initial.tree_accession,
            "repeat_representative_pre_KH1_aa": r.pre_KH1_length,
            "tree_isoform_pre_KH1_aa": tree_isoform.pre_KH1_length,
            "repeat_representative_QC_issues": r.repeat_region_qc_issues,
            "same_gene_QC_passing_isoforms": len(same_gene_rescue),
            "other_gene_QC_passing_candidates": len(alternatives),
            "other_gene_QC_passing_on_FLK_side": len(in_flk),
            "other_gene_FLK_side_accessions": ";".join(in_flk.tree_accession),
            "in_261_tip_FLK_side": initial.in_all_gene_FLK_side,
            "in_102_tip_FLK_side": initial.in_focus_FLK_side,
            "in_occupancy_sensitivity_FLK_side": sen.in_occupancy_FLK_side,
            "FLK_distance_advantage": initial.FLK_distance_advantage,
            "uniparc_interpretation": uni.interpretation,
            "resolution_category": category,
            "one_to_one_orthology_verified": False,
        })
    out = pd.DataFrame(rows).sort_values("species")
    if len(out) != 14:
        raise AssertionError(f"Expected 14 QC failures, got {len(out)}")
    out.to_csv(OUT / "qc_failed_species_resolution.tsv", sep="\t", index=False)
    print(out.resolution_category.value_counts().to_string())
    print(f"wrote {OUT / 'qc_failed_species_resolution.tsv'}")
    return out


if __name__ == "__main__":
    main()
