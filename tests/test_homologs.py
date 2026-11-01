"""Homolog curation and repeat-comparison invariants."""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import homologs as hom
from flk_repeat import blosum62 as bl
from flk_repeat.paths import RAW_HOMOLOG_CANDIDATES_FASTA, HOMOLOG_PROVENANCE_TSV, CURATED_HOMOLOG_FASTA


def test_provenance_table_covers_every_raw_candidate():
    raw = hom.parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    prov = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    assert set(prov.accession) == set(raw.keys())


def test_included_count_in_target_range():
    prov = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    n_included = int(prov.included.sum())
    assert 15 <= n_included <= 30


def test_reference_is_q9sr13_and_included():
    prov = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    ref = prov[prov.orthology_status == "REFERENCE"]
    assert len(ref) == 1
    assert ref.accession.iloc[0] == "Q9SR13"
    assert bool(ref.included.iloc[0]) is True


def test_no_confirmed_ortholog_label_anywhere():
    prov = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    assert "CONFIRMED_ORTHOLOG" not in set(prov.orthology_status)


def test_every_excluded_row_has_a_reason():
    prov = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    excluded = prov[~prov.included]
    assert len(excluded) > 0
    assert excluded.inclusion_or_exclusion_reason.notna().all()
    assert (excluded.inclusion_or_exclusion_reason.str.len() > 0).all()


def test_sequence_hashes_match_raw_fasta():
    from flk_repeat.provenance import sha256_of_sequence
    raw = hom.parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    prov = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    for _, row in prov.iterrows():
        seq = raw[row.accession]
        assert len(seq) == row.length, f"{row.accession}: length mismatch"
        assert sha256_of_sequence(seq) == row.sha256, f"{row.accession}: hash mismatch"


def test_needleman_wunsch_recovers_known_arabis_alpina_windows():
    """Regression check against the prior repository's own audited Clustal-
    Omega-derived values for Arabis alpina (independently cross-validated
    when re-fetched with a real accession, see docs/METHODS.md)."""
    raw = hom.parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    athal = raw["Q9SR13"]
    arabis = raw["A0A087H6X3"]
    athal_region = athal[:185]  # before KH1 at 186
    arabis_region = arabis[:184]  # before KH1 at 185 (see homolog_provenance.tsv)
    ref_aln, query_aln = hom.needleman_wunsch(athal_region, arabis_region)
    col_of_pos = hom.ungapped_position_to_column(ref_aln)
    for start, expected in [(51, "LEAQQYEV"), (63, "LEAQQYGV"), (75, "LEAQQFGV")]:
        cols = [col_of_pos[start + i] for i in range(8)]
        window = "".join(query_aln[c] for c in cols)
        assert window == expected


def test_curated_fasta_is_ungapped_and_included_only():
    raw = hom.parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    prov_df = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t", dtype={"kh1_start_1based": "Int64"})
    rows = prov_df.to_dict(orient="records")
    for r in rows:
        r["included"] = bool(r["included"])
    curated = hom.parse_fasta(CURATED_HOMOLOG_FASTA)
    n_included = sum(r["included"] for r in rows)
    assert len(curated) == n_included
    for header, seq in curated.items():
        assert "-" not in seq


def test_blosum62_self_similarity_is_one():
    assert bl.normalized_similarity("LEPQQYEV", "LEPQQYEV") == 1.0


def test_blosum62_gap_scores_worse_than_any_real_pair():
    assert bl.score_pair("-", "L") == bl.GAP_SCORE
    assert bl.GAP_SCORE < bl.BLOSUM62[("L", "L")]


def test_needleman_wunsch_identical_sequences_gapless():
    ref_aln, query_aln = hom.needleman_wunsch("LEPQQYEV", "LEPQQYEV")
    assert ref_aln == query_aln == "LEPQQYEV"


def test_needleman_wunsch_handles_deletion():
    # query is ref with residue 5 (Q) deleted
    ref_aln, query_aln = hom.needleman_wunsch("LEPQQYEV", "LEPQYEV")
    assert ref_aln.replace("-", "") == "LEPQQYEV"
    assert query_aln.replace("-", "") == "LEPQYEV"
    assert len(ref_aln) == len(query_aln)
    assert "-" in query_aln
