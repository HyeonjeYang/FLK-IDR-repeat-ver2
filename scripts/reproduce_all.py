#!/usr/bin/env python3
"""One-command reproduction of every table and figure in this repository,
from repository-local data only (no network access required).

    python scripts/reproduce_all.py

Steps: validate source files and sequence identity/hashes -> regenerate
results/*.tsv -> run the composition-preserving null models -> regenerate
figures/*.png -> run basic consistency/invariant checks. Exits non-zero on
any failure.
"""
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)
    return condition


def step_validate_sources():
    print("\n== Step 1: validate required source files ==")
    from flk_repeat.paths import (FLK_SEQUENCE_JSON, FLK_IUPRED_JSON, FLK_PROVENANCE_JSON,
                                   RAW_HOMOLOG_CANDIDATES_FASTA, HOMOLOG_PROVENANCE_TSV,
                                   LEGACY_RAW_HOMOLOG_FASTA, DATA_DIR)
    for label, path in [
        ("data/sequence/Q9SR13.json", FLK_SEQUENCE_JSON),
        ("data/sequence/Q9SR13_iupred2a.json", FLK_IUPRED_JSON),
        ("data/provenance/flk_sequence_provenance.json", FLK_PROVENANCE_JSON),
        ("data/raw/homolog_candidates_raw.fasta", RAW_HOMOLOG_CANDIDATES_FASTA),
        ("data/provenance/homolog_provenance.tsv", HOMOLOG_PROVENANCE_TSV),
        ("data/homologs/raw/flk_homologs_raw_aligned.fasta (legacy)", LEGACY_RAW_HOMOLOG_FASTA),
        ("data/validation/flk_full_length_clustalo.fasta", DATA_DIR / "validation" / "flk_full_length_clustalo.fasta"),
        ("data/validation/alignment_validation_metadata.tsv", DATA_DIR / "validation" / "alignment_validation_metadata.tsv"),
        ("data/expanded/background/athaliana_UP000006548_regions.tsv.gz",
         DATA_DIR / "expanded" / "background" / "athaliana_UP000006548_regions.tsv.gz"),
        ("data/expanded/ensembl/compara_plants_AT3G04610_orthologues.json",
         DATA_DIR / "expanded" / "ensembl" / "compara_plants_AT3G04610_orthologues.json"),
        ("data/expanded/verification/reverse_blast_athaliana.json",
         DATA_DIR / "expanded" / "verification" / "reverse_blast_athaliana.json"),
        ("data/expanded/verification/reverse_blast_athaliana_subjects.gp",
         DATA_DIR / "expanded" / "verification" / "reverse_blast_athaliana_subjects.gp"),
        ("data/expanded/sequences/candidate_hits.gp",
         DATA_DIR / "expanded" / "sequences" / "candidate_hits.gp"),
        ("data/expanded/sequences/representative_gene_isoforms.gp",
         DATA_DIR / "expanded" / "sequences" / "representative_gene_isoforms.gp"),
        ("data/expanded/paralogs/athaliana_KH_reviewed.tsv",
         DATA_DIR / "expanded" / "paralogs" / "athaliana_KH_reviewed.tsv"),
        ("data/expanded/verification/uniparc_crossrefs.json",
         DATA_DIR / "expanded" / "verification" / "uniparc_crossrefs.json"),
    ]:
        check(f"{label} exists", path.exists())


def step_validate_sequence_identity():
    print("\n== Step 2: validate FLK sequence identity/hashes ==")
    from flk_repeat import sequence as seqmod
    rec = seqmod.load_flk_record()
    check("accession == Q9SR13", rec["accession"] == "Q9SR13")
    check("length == 577", rec["length"] == 577)
    check("sequence SHA256 matches stored provenance record",
          rec["sha256"] == rec["provenance"]["sequence_sha256_uppercase"])
    check("record is UniProtKB reviewed", rec["reviewed"] is True)
    scores = seqmod.load_iupred_scores(rec["length"])
    check("IUPred2A cache is length-aligned to the sequence", len(scores) == rec["length"])
    return rec


def step_build_results(flk_seq):
    print("\n== Step 3: regenerate results/*.tsv ==")
    import analyze_flk_architecture
    import analyze_repeat_conservation
    import run_composition_null

    arch_out = analyze_flk_architecture.main()
    conservation_out = analyze_repeat_conservation.main(arch_out["starts"], arch_out["kh1_start_1based"])
    null_out = run_composition_null.main()
    import validate_figure_statistics
    validate_figure_statistics.main()

    print("\n== Step 3b: regenerate results/flk_alignment_validation*.tsv "
          "(from the stored, immutable MSA -- no network access) ==")
    import run_alignment_validation
    validation_out = run_alignment_validation.main()

    import analyze_curated_reference_free
    analyze_curated_reference_free.main()

    import build_idr_background, analyze_curated_chemistry
    build_idr_background.main()
    analyze_curated_chemistry.main()

    import analyze_compara_pilot
    analyze_compara_pilot.main()
    import analyze_expanded_reciprocal
    analyze_expanded_reciprocal.main()
    import audit_expanded_isoforms, triage_expanded_gene_candidates
    audit_expanded_isoforms.main()
    triage_expanded_gene_candidates.main()
    import analyze_expanded_phylogeny, check_expanded_phylogeny_sensitivity
    analyze_expanded_phylogeny.main(bootstraps=400)
    check_expanded_phylogeny_sensitivity.main(bootstraps=400)
    import audit_uniparc_verification
    audit_uniparc_verification.main()
    import summarize_expanded_qc_failures
    summarize_expanded_qc_failures.main()
    import analyze_revision_claims, analyze_nonoverlap_null, audit_flk_codon_splice
    analyze_revision_claims.main()
    analyze_nonoverlap_null.main()
    audit_flk_codon_splice.main()

    return arch_out, conservation_out, null_out, validation_out


def step_build_figures():
    print("\n== Step 4: regenerate figures/ ==")
    import make_fig1_architecture, make_fig2_conservation, make_fig3_null, make_figS1_alignment_validation
    make_fig1_architecture.main()
    make_fig2_conservation.main()
    make_fig3_null.main()
    make_figS1_alignment_validation.main()
    import make_figS5_expanded_phylogeny
    make_figS5_expanded_phylogeny.main()


def step_consistency_checks(arch_out, conservation_out, null_out, validation_out):
    print("\n== Step 5: basic consistency/invariant checks ==")
    import pandas as pd
    from flk_repeat.paths import (RESULTS_DIR, FIGURES_DIR, CURATED_HOMOLOG_FASTA,
                                   RAW_HOMOLOG_CANDIDATES_FASTA, HOMOLOG_PROVENANCE_TSV)
    from flk_repeat import homologs as hom

    for fname in ["fig1_flk_architecture", "fig2_repeat_conservation", "fig3_composition_null",
                  "figS1_alignment_validation", "figS2_reference_free_curated",
                  "figS3_curated_chemistry", "figS4_compara_pilot",
                  "figS5_expanded_phylogeny_qc"]:
        p = FIGURES_DIR / f"{fname}.png"
        check(f"{fname}.png exists and is non-trivially sized",
              p.exists() and p.stat().st_size > 1000)

    for fname in ["flk_sequence_provenance.tsv", "flk_repeat_coordinates.tsv",
                  "flk_window_composition.tsv", "flk_domain_annotation.tsv",
                  "flk_homolog_repeat_comparison.tsv", "flk_null_model_summary.tsv",
                  "flk_null_model_histogram.tsv", "flk_alignment_validation.tsv",
                  "flk_alignment_validation_summary.tsv"]:
        p = RESULTS_DIR / fname
        try:
            df = pd.read_csv(p, sep="\t")
            check(f"results/{fname} parses and is non-empty", len(df) > 0)
        except Exception as e:
            check(f"results/{fname} parses and is non-empty ({e})", False)

    coords = pd.read_csv(RESULTS_DIR / "flk_repeat_coordinates.tsv", sep="\t")
    check("exactly 3 LEPQQYEV repeats reported", len(coords) == 3)
    check("all reported repeats are exactly LEPQQYEV",
          all(coords.sequence == "LEPQQYEV"))

    comparison = pd.read_csv(RESULTS_DIR / "flk_homolog_repeat_comparison.tsv", sep="\t")
    check("no homolog-comparison row is labelled CONFIRMED_ORTHOLOG",
          "CONFIRMED_ORTHOLOG" not in set(comparison.orthology_status))
    provenance = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    n_included = int(provenance.included.sum())
    check("homolog comparison covers exactly (included candidates - 1 reference) x 3 repeats",
          len(comparison) == (n_included - 1) * 3)
    check("curated set has 15-30 included candidates", 15 <= n_included <= 30)

    reference_free = pd.read_csv(RESULTS_DIR / "expanded" / "curated_reference_free_repeats.tsv", sep="\t")
    check("reference-free survey covers each included candidate exactly once",
          len(reference_free) == n_included and reference_free.accession.is_unique)
    check("reference-free approximate-copy counts increase or stay constant with tolerance",
          (reference_free.best_8mer_2_mismatch_nonoverlapping_copies >=
           reference_free.best_8mer_1_mismatch_nonoverlapping_copies).all())
    chemistry = pd.read_csv(RESULTS_DIR / "expanded" / "curated_pre_kh1_chemistry.tsv", sep="\t")
    check("chemistry survey covers the same candidates as reference-free survey",
          set(chemistry.accession) == set(reference_free.accession) and chemistry.accession.is_unique)
    background = pd.read_csv(RESULTS_DIR / "expanded" / "athaliana_idr_background_features.tsv", sep="\t")
    check("disordered-region chemistry background contains 13,253 regions", len(background) == 13253)
    compara_pilot = pd.read_csv(RESULTS_DIR / "expanded" / "compara_pilot_repeats.tsv", sep="\t")
    check("Compara pilot includes multiple monocots and other angiosperms",
          (compara_pilot.lineage == "Monocots").sum() >= 3 and
          (compara_pilot.lineage != "Monocots").sum() >= 3)
    check("Compara pilot candidates remain explicitly unverified",
          set(compara_pilot.verification_status) == {"COMPARA_ONLY_UNVERIFIED"})
    reciprocal = pd.read_csv(RESULTS_DIR / "expanded" / "reciprocal_blast_audit.tsv", sep="\t")
    check("expanded RefSeq reciprocal audit covers 40 distinct representatives",
          len(reciprocal) == 40 and reciprocal.query_accession.is_unique)
    check("all 40 expanded RefSeq top hits map to FLK GeneID 819619",
          reciprocal.top_hit_is_flk_gene.all() and
          set(reciprocal.top_athaliana_gene_id.astype(str)) == {"819619"})
    check("26 expanded RefSeq representatives pass repeat-region QC",
          int(reciprocal.passes_repeat_region_qc.sum()) == 26)
    isoforms = pd.read_csv(RESULTS_DIR / "expanded" / "selected_gene_isoform_qc.tsv", sep="\t")
    check("75 cached isoforms of 40 selected genes audited",
          len(isoforms) == 75 and isoforms.species.nunique() == 40)
    failed_species = set(reciprocal.loc[~reciprocal.passes_repeat_region_qc, "species"])
    check("none of 14 QC-failing initial genes has a passing same-gene isoform",
          not isoforms.loc[isoforms.species.isin(failed_species),
                           "passes_repeat_region_qc"].any())
    phylo_audit = pd.read_csv(RESULTS_DIR / "expanded" / "expanded_phylogeny_audit.tsv", sep="\t")
    initial = phylo_audit[phylo_audit.is_initial_selected_gene]
    check("40 initial genes lie on FLK side in both KH-core trees",
          len(initial) == 40 and initial.in_all_gene_FLK_side.all() and
          initial.in_focus_FLK_side.all())
    check("initial genes are closer to A. thaliana FLK than listed KH paralogs",
          (initial.FLK_distance_advantage > 0).all())
    sensitivity = pd.read_csv(RESULTS_DIR / "expanded" / "expanded_phylogeny_sensitivity.tsv", sep="\t")
    check("all 40 initial genes remain on FLK side in occupancy-filter sensitivity tree",
          len(sensitivity) == 40 and sensitivity.in_occupancy_FLK_side.all())
    uniparc = pd.read_csv(RESULTS_DIR / "expanded" / "uniparc_sequence_verification.tsv", sep="\t")
    check("40 RefSeq representatives have explicit UniParc cross-reference outcomes",
          len(uniparc) == 40 and uniparc.accession.is_unique)
    resolution = pd.read_csv(RESULTS_DIR / "expanded" / "qc_failed_species_resolution.tsv", sep="\t")
    check("14 repeat-region QC failures have species-level resolution audit",
          len(resolution) == 14 and resolution.species.is_unique and
          not resolution.one_to_one_orthology_verified.any())
    qc_repeat = pd.read_csv(RESULTS_DIR / "expanded" / "qc_passed_repeat_descriptive.tsv", sep="\t")
    check("26 QC-passing species have a descriptive repeat scan",
          len(qc_repeat) == 26 and qc_repeat.species.is_unique and
          (qc_repeat.LEPQQYEV_exact_copies == 0).all())
    close = pd.read_csv(RESULTS_DIR / "expanded" / "close_relative_retention.tsv", sep="\t")
    controls = pd.read_csv(RESULTS_DIR / "expanded" / "close_relative_spacing_controls.tsv", sep="\t")
    check("close-relative repeat/flank counts and shared-position controls are present",
          list(close.repeat_differences) == [1, 1, 0] and
          list(close.flank_differences) == [26, 18, 16] and
          len(controls) == 56 and
          (controls.pairwise_total_differences_three_species > 2).all() and
          (controls.MSA_total_differences_three_species > 2).all())
    bg = pd.read_csv(RESULTS_DIR / "expanded" / "athaliana_idr_150aa_window_repeats.tsv", sep="\t")
    check("length-matched IDR window background is complete",
          len(bg) == 883 and int((bg.T8_exact_overlapping >= 3).sum()) == 121 and
          int((bg.T8_exact_nonoverlapping >= 3).sum()) == 93)
    no_null = pd.read_csv(RESULTS_DIR / "expanded" / "nonoverlap_null_summary.tsv", sep="\t")
    check("independent non-overlap null uses two million total draws",
          len(no_null) == 2 and (no_null.N == 1_000_000).all() and
          (no_null.nonoverlap_exceedances == 0).all())

    # Curated FASTA must be exactly reproducible from the raw FASTA + provenance table.
    from analyze_repeat_conservation import load_provenance_rows
    raw_seqs = hom.parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    prov_rows = load_provenance_rows()
    tmp_path = RESULTS_DIR / "_tmp_curated_check.fasta"
    hom.build_curated_fasta(raw_seqs, prov_rows, tmp_path)
    regenerated = tmp_path.read_text(encoding="utf-8")
    committed = CURATED_HOMOLOG_FASTA.read_text(encoding="utf-8")
    tmp_path.unlink()
    check("curated FASTA is byte-identical when regenerated from raw FASTA + provenance table",
          regenerated == committed)

    null_summary = pd.read_csv(RESULTS_DIR / "flk_null_model_summary.tsv", sep="\t")
    check("null model summary has 8 rows (3 windows global + 5 segmental phase-offsets)",
          len(null_summary) == 8)
    check("null model always used >= 1,000,000 permutations", (null_summary.N_permutations >= 1_000_000).all())
    check("empirical p-value 95% CIs are present, in [0,1], and well-ordered (lower <= upper)",
          ((null_summary.empirical_p_ci_lower >= 0) & (null_summary.empirical_p_ci_upper <= 1) &
           (null_summary.empirical_p_ci_lower <= null_summary.empirical_p_ci_upper)).all())
    check("empirical p-values are all in (0, 1]", ((null_summary.empirical_p > 0) & (null_summary.empirical_p <= 1)).all())
    figure3_audit = pd.read_csv(RESULTS_DIR / "expanded" / "figure3_statistical_audit.tsv", sep="\t")
    check("all eight Figure 3 histogram-to-summary statistical audits pass",
          len(figure3_audit) == 8 and figure3_audit.all_checks_pass.all())

    segmental = null_summary[null_summary.null_model == "segmental"]
    check("segmental null was run at exactly 5 phase offsets (0,5,10,15,20)",
          sorted(segmental.phase_offset.astype(int)) == [0, 5, 10, 15, 20])
    check("segmental-null result is robust to block-grid phase (all p_emp_T8 < 0.01)",
          (segmental.empirical_p < 0.01).all())

    window_comp = pd.read_csv(RESULTS_DIR / "flk_window_composition.tsv", sep="\t")
    primary_len = int(window_comp.loc[window_comp.window == "primary_1_150", "length"].iloc[0])
    check("primary window (1-150) length is 150", primary_len == 150)
    sens107_len = int(window_comp.loc[window_comp.window == "sensitivity_1_107", "length"].iloc[0])
    sens182_len = int(window_comp.loc[window_comp.window == "sensitivity_1_182", "length"].iloc[0])
    check("sensitivity window (1-107) length is 107", sens107_len == 107)
    check("sensitivity window (1-182) length is 182", sens182_len == 182)

    val_detail = pd.read_csv(RESULTS_DIR / "flk_alignment_validation.tsv", sep="\t")
    provenance = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    n_included = int(provenance.included.sum())
    check("alignment validation covers exactly (included candidates - 1 reference) x 3 repeats",
          len(val_detail) == (n_included - 1) * 3)
    allowed_status = {"EXACT_AGREEMENT", "NONMATERIAL_DIFFERENCE", "MATERIAL_DISAGREEMENT"}
    check("alignment validation status values are all recognized",
          set(val_detail.alignment_validation_status) <= allowed_status)

    val_summary = pd.read_csv(RESULTS_DIR / "flk_alignment_validation_summary.tsv", sep="\t")
    overall_row = val_summary[val_summary.curation_group == "ALL"].iloc[0]
    check("alignment validation summary row totals match the detail table",
          int(overall_row.n_repeat_positions_total) == len(val_detail))
    check("policy invariant: n_alignment_sensitive == n_material_disagreement in every group",
          (val_summary.n_alignment_sensitive == val_summary.n_material_disagreement).all())


def main():
    print(f"Repository root: {REPO_ROOT}")
    step_validate_sources()
    rec = step_validate_sequence_identity()
    arch_out, conservation_out, null_out, validation_out = step_build_results(rec["sequence"])
    step_build_figures()
    step_consistency_checks(arch_out, conservation_out, null_out, validation_out)

    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"REPRODUCE_ALL: FAILED ({len(FAILURES)} check(s) failed)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("REPRODUCE_ALL: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
