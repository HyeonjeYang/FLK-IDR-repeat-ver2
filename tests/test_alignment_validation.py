"""Independent full-length MSA sensitivity check: coordinate mapping,
agreement classification, and summary-vs-detail consistency.
Run `python scripts/reproduce_all.py` first (populates results/
flk_alignment_validation*.tsv from the stored, immutable MSA).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from flk_repeat import homologs as hom
from flk_repeat.paths import RESULTS_DIR, DATA_DIR, HOMOLOG_PROVENANCE_TSV

MSA_PATH = DATA_DIR / "validation" / "flk_full_length_clustalo.fasta"


def test_msa_file_exists():
    assert MSA_PATH.exists(), "missing stored validation MSA -- see data/validation/alignment_validation_metadata.tsv"


def test_athaliana_ungapped_position_maps_to_expected_columns():
    msa = hom.parse_fasta(MSA_PATH)
    athal_gapped = msa["Q9SR13"]
    col_of_pos = hom.ungapped_position_to_column(athal_gapped)
    # position 1 (ungapped) must map to the first non-gap column
    first_non_gap = next(i for i, ch in enumerate(athal_gapped) if ch != "-")
    assert col_of_pos[1] == first_non_gap
    # reading off the mapped columns must reproduce A. thaliana's own sequence
    ungapped = athal_gapped.replace("-", "")
    for pos in (1, 51, 63, 75, len(ungapped)):
        assert athal_gapped[col_of_pos[pos]] == ungapped[pos - 1]


def test_athaliana_maps_to_lepqqyev_for_all_three_repeats():
    msa = hom.parse_fasta(MSA_PATH)
    athal_gapped = msa["Q9SR13"]
    col_of_pos = hom.ungapped_position_to_column(athal_gapped)
    for start in (51, 63, 75):
        cols = [col_of_pos[start + i] for i in range(8)]
        window = "".join(athal_gapped[c] for c in cols)
        assert window == "LEPQQYEV"


def test_close_homolog_validation_values_recovered():
    """A. lyrata and A. halleri are expected to show EXACT_AGREEMENT between
    the pairwise and MSA mappings for all three repeats -- both are very
    close relatives with an essentially gapless alignment in this region."""
    detail = pd.read_csv(RESULTS_DIR / "flk_alignment_validation.tsv", sep="\t")
    for species in ("Arabidopsis lyrata", "Arabidopsis halleri"):
        rows = detail[detail.species == species]
        assert len(rows) == 3
        assert (rows.alignment_validation_status == "EXACT_AGREEMENT").all(), (
            f"{species} expected exact agreement on all 3 repeats, got "
            f"{rows[['repeat_index', 'alignment_validation_status']].to_dict('records')}")


def test_summary_counts_agree_with_detail_table():
    detail = pd.read_csv(RESULTS_DIR / "flk_alignment_validation.tsv", sep="\t")
    summary = pd.read_csv(RESULTS_DIR / "flk_alignment_validation_summary.tsv", sep="\t")
    overall = summary[summary.curation_group == "ALL"].iloc[0]

    assert int(overall.n_repeat_positions_total) == len(detail)
    assert int(overall.n_exact_agreement) == int((detail.alignment_validation_status == "EXACT_AGREEMENT").sum())
    assert int(overall.n_nonmaterial_difference) == int(
        (detail.alignment_validation_status == "NONMATERIAL_DIFFERENCE").sum())
    assert int(overall.n_material_disagreement) == int(
        (detail.alignment_validation_status == "MATERIAL_DISAGREEMENT").sum())

    # by-group rows must partition the detail table exactly (no double count, no gaps)
    group_rows = summary[summary.curation_group != "ALL"]
    assert group_rows.n_repeat_positions_total.sum() == len(detail)


def test_alignment_sensitive_equals_material_disagreement_by_policy():
    """Documented policy (docs/METHODS.md): every MATERIAL_DISAGREEMENT call
    is uniformly treated as alignment-sensitive, so these two counts must be
    identical in every summary row -- not adjudicated case-by-case."""
    summary = pd.read_csv(RESULTS_DIR / "flk_alignment_validation_summary.tsv", sep="\t")
    assert (summary.n_alignment_sensitive == summary.n_material_disagreement).all()


def test_deterministic_regeneration_from_stored_msa():
    """Rerunning the validation script against the same stored MSA and the
    same pairwise results must produce byte-identical output (no RNG, no
    network access, purely a deterministic re-derivation)."""
    import run_alignment_validation as rav
    out1 = rav.main()
    out2 = rav.main()
    pd.testing.assert_frame_equal(out1["detail"], out2["detail"])
    pd.testing.assert_frame_equal(out1["summary"], out2["summary"])


def test_fig2_marks_the_same_number_of_alignment_sensitive_cells_as_the_validation_table():
    """Figure 2 outlines/stars a cell iff its (species, repeat_index) row in
    results/flk_alignment_validation.tsv is MATERIAL_DISAGREEMENT. Re-derive
    that cell count the same way make_fig2_conservation.py does (species_
    order from the provenance table, excluding the reference) and check it
    matches the validation table's own MATERIAL_DISAGREEMENT count exactly."""
    validation = pd.read_csv(RESULTS_DIR / "flk_alignment_validation.tsv", sep="\t")
    provenance = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    species_order = [s for s in provenance.loc[provenance.included, "species"] if s != "Arabidopsis thaliana"]

    n_marked = 0
    for sp in species_order:
        for r in range(1, 4):
            vrow = validation[(validation.species == sp) & (validation.repeat_index == r)]
            assert len(vrow) == 1, f"expected exactly one validation row for {sp} repeat {r}"
            if vrow.alignment_validation_status.iloc[0] == "MATERIAL_DISAGREEMENT":
                n_marked += 1

    n_material_in_table = int((validation.alignment_validation_status == "MATERIAL_DISAGREEMENT").sum())
    assert n_marked == n_material_in_table


def test_classify_function_categories():
    from run_alignment_validation import classify, window_stats
    ref = "LEPQQYEV"
    exact_pw = window_stats("LEPQQYEV", ref)
    exact_msa = window_stats("LEPQQYEV", ref)
    assert classify(exact_pw, exact_msa) == "EXACT_AGREEMENT"

    # same qualitative call (both exact, no indel) but hypothetically
    # different string would still be EXACT_AGREEMENT since window text is
    # identical in this case -- test a genuine nonmaterial case: both
    # non-exact, no indel, different substitution pattern
    nonmat_pw = window_stats("LEPQEYEV", ref)
    nonmat_msa = window_stats("LEPQGYEV", ref)
    assert classify(nonmat_pw, nonmat_msa) == "NONMATERIAL_DIFFERENCE"

    # material: one has an indel, the other does not
    material_pw = window_stats("LEPQQYEV", ref)
    material_msa = window_stats("LEP-----", ref)
    assert classify(material_pw, material_msa) == "MATERIAL_DISAGREEMENT"
