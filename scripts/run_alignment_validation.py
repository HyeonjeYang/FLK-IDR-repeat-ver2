#!/usr/bin/env python3
"""Independent full-length MSA sensitivity check on the primary KH1-anchored
pairwise repeat mapping (SECONDARY / validation analysis -- see docs/
METHODS.md, "Independent full-length MSA sensitivity check").

This script does NOT run Clustal Omega. It validates and re-derives from
the exact, immutable alignment already stored in
data/validation/flk_full_length_clustalo.fasta (see data/validation/
alignment_validation_metadata.tsv for how and when that file was produced)
and compares it against the primary pairwise-derived repeat windows already
in results/flk_homolog_repeat_comparison.tsv. No motif re-search or local
optimization is performed: A. thaliana's three reference repeat coordinates
(51-58, 63-70, 75-82) are mapped through the MSA's alignment columns exactly
once, and whatever residues land in those columns for each species are
reported -- never the locally best-matching 8-mer.

Writes:
    results/flk_alignment_validation.tsv
    results/flk_alignment_validation_summary.tsv
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import homologs as hom
from flk_repeat.architecture import MOTIF
from flk_repeat.blosum62 import alignment_score, self_score
from flk_repeat.paths import RESULTS_DIR, HOMOLOG_PROVENANCE_TSV, DATA_DIR, FLK_PROVENANCE_JSON

VALIDATION_DIR = DATA_DIR / "validation"
MSA_PATH = VALIDATION_DIR / "flk_full_length_clustalo.fasta"
REPEAT_STARTS = [51, 63, 75]

CURATION_GROUP = {
    "REFERENCE": "reference",
    "HIGH_CONFIDENCE_ORTHOLOG": "close_arabidopsis_or_brassicaceae",
    "SUPPORTED_ORTHOLOG_CANDIDATE": "distant_eudicot",
    "UNCERTAIN": "rice_uncertain",
}
# Finer split for the summary-by-group table: the closest 6 (genus
# Arabidopsis + Arabis alpina) vs the remaining Brassicaceae, matching the
# ordering already used throughout data/provenance/homolog_provenance.tsv.
CLOSE_ARABIDOPSIS_SPECIES = {
    "Arabidopsis lyrata", "Arabidopsis halleri", "Arabidopsis arenosa",
    "Arabidopsis suecica", "Arabis alpina",
}


def window_stats(window: str, ref: str = MOTIF) -> dict:
    has_indel = "-" in window
    exact_identity = (not has_indel) and window == ref
    return {
        "window": window,
        "exact_identity": exact_identity,
        "has_indel": has_indel,
        "blosum62_similarity": max(0.0, alignment_score(window, ref) / self_score(ref)),
    }


def classify(pairwise: dict, msa: dict) -> str:
    if pairwise["window"] == msa["window"]:
        return "EXACT_AGREEMENT"
    if (pairwise["exact_identity"] == msa["exact_identity"] and
            pairwise["has_indel"] == msa["has_indel"]):
        return "NONMATERIAL_DIFFERENCE"
    return "MATERIAL_DISAGREEMENT"


def curation_group(species: str, status: str) -> str:
    if status != "HIGH_CONFIDENCE_ORTHOLOG":
        return CURATION_GROUP.get(status, status)
    return "close_arabidopsis" if species in CLOSE_ARABIDOPSIS_SPECIES else "other_brassicaceae"


def kh_domain_qc(msa: dict, athal_col_of_pos: dict, kh_domains: list) -> pd.DataFrame:
    """Gap fraction of each included species across each KH-domain's aligned
    columns -- a sanity check that the structured core is not obviously
    misaligned (the validation's purpose is the N-terminal repeat, not KH
    domain evolution, so this is reported but not analyzed further)."""
    rows = []
    for kh in kh_domains:
        cols = [athal_col_of_pos[p] for p in range(kh["start"], kh["end"] + 1)]
        for acc, gapped in msa.items():
            if acc == "Q9SR13":
                continue
            window = "".join(gapped[c] for c in cols)
            rows.append({"kh_domain": kh["name"], "accession": acc,
                         "gap_fraction": window.count("-") / len(window)})
    return pd.DataFrame(rows)


def main():
    import json
    msa = hom.parse_fasta(MSA_PATH)
    provenance = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    included = provenance[provenance.included]

    assert set(msa.keys()) == set(included.accession), (
        "stored MSA accessions do not match the currently included curated set -- "
        "the validation alignment must be regenerated (see data/validation/"
        "alignment_validation_metadata.tsv) before this comparison is meaningful")

    athal_gapped = msa["Q9SR13"]
    col_of_pos = hom.ungapped_position_to_column(athal_gapped)
    pairwise = pd.read_csv(RESULTS_DIR / "flk_homolog_repeat_comparison.tsv", sep="\t")

    rows = []
    for repeat_idx, start in enumerate(REPEAT_STARTS, start=1):
        cols = [col_of_pos[start + i] for i in range(len(MOTIF))]
        for _, prow in included.iterrows():
            acc = prow.accession
            if acc == "Q9SR13":
                continue
            msa_window = "".join(msa[acc][c] for c in cols)
            msa_stats = window_stats(msa_window)

            pw = pairwise[(pairwise.accession == acc) & (pairwise.repeat_index == repeat_idx)]
            pw_window = pw.aligned_sequence.iloc[0]
            pw_stats = window_stats(pw_window)

            status = classify(pw_stats, msa_stats)
            rows.append({
                "species": prow.species,
                "accession": acc,
                "orthology_status": prow.orthology_status,
                "curation_group": curation_group(prow.species, prow.orthology_status),
                "repeat_index": repeat_idx,
                "pairwise_window": pw_stats["window"],
                "msa_window": msa_stats["window"],
                "exact_window_match": pw_stats["window"] == msa_stats["window"],
                "pairwise_exact_identity": pw_stats["exact_identity"],
                "msa_exact_identity": msa_stats["exact_identity"],
                "pairwise_has_indel": pw_stats["has_indel"],
                "msa_has_indel": msa_stats["has_indel"],
                "pairwise_blosum62_similarity": pw_stats["blosum62_similarity"],
                "msa_blosum62_similarity": msa_stats["blosum62_similarity"],
                "alignment_validation_status": status,
                "notes": ("" if status != "MATERIAL_DISAGREEMENT" else
                          "treated as alignment-sensitive: not used to assert a specific "
                          "retained/lost call for this species/repeat position (see docs/METHODS.md)"),
            })

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(RESULTS_DIR / "flk_alignment_validation.tsv", sep="\t", index=False)

    def summarize(df: pd.DataFrame) -> dict:
        n = len(df)
        counts = df.alignment_validation_status.value_counts()
        n_exact = int(counts.get("EXACT_AGREEMENT", 0))
        n_material = int(counts.get("MATERIAL_DISAGREEMENT", 0))
        return {
            "n_repeat_positions_total": n,
            "n_exact_agreement": n_exact,
            "n_nonmaterial_difference": int(counts.get("NONMATERIAL_DIFFERENCE", 0)),
            "n_material_disagreement": n_material,
            # policy (documented in docs/METHODS.md): every MATERIAL_DISAGREEMENT
            # call is uniformly treated as alignment-sensitive, not adjudicated
            # case-by-case, so this count is always equal to n_material_disagreement.
            "n_alignment_sensitive": n_material,
            "fraction_exact_agreement": round(n_exact / n, 3) if n else float("nan"),
        }

    overall = {"curation_group": "ALL", "n_species": detail_df.accession.nunique(), **summarize(detail_df)}
    by_group = []
    for group, gdf in detail_df.groupby("curation_group"):
        by_group.append({"curation_group": group, "n_species": gdf.accession.nunique(), **summarize(gdf)})

    summary_df = pd.DataFrame([overall] + sorted(by_group, key=lambda r: r["curation_group"]))
    summary_df.to_csv(RESULTS_DIR / "flk_alignment_validation_summary.tsv", sep="\t", index=False)

    # KH-domain alignment QC (item 7): sanity check only, printed not stored.
    flk_prov = json.loads(FLK_PROVENANCE_JSON.read_text(encoding="utf-8"))
    kh_qc = kh_domain_qc(msa, col_of_pos, flk_prov["kh_domains_1based_inclusive"])
    print("KH-domain alignment QC (mean gap fraction across included species' aligned columns):")
    print(kh_qc.groupby("kh_domain").gap_fraction.mean().to_string())
    high_gap = kh_qc[kh_qc.gap_fraction > 0.5]
    if len(high_gap):
        print(f"  NOTE: {len(high_gap)} (species, KH domain) pairs have >50% gaps -- inspect before trusting:")
        print(high_gap.to_string(index=False))
    else:
        print("  all included species have <=50% gaps across all three KH domains -- "
              "structured core aligns sensibly, no red flag for the N-terminal validation above.")

    print(f"\nOverall: {overall['n_exact_agreement']}/{overall['n_repeat_positions_total']} exact agreement "
          f"({overall['fraction_exact_agreement']:.1%}), "
          f"{overall['n_nonmaterial_difference']} nonmaterial, "
          f"{overall['n_material_disagreement']} material disagreements (-> alignment-sensitive)")
    for row in by_group:
        print(f"  {row['curation_group']:<28} exact={row['n_exact_agreement']}/{row['n_repeat_positions_total']} "
              f"({row['fraction_exact_agreement']:.1%})  material={row['n_material_disagreement']}")

    return {"detail": detail_df, "summary": summary_df}


if __name__ == "__main__":
    main()
