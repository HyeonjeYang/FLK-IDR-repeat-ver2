"""Sequence identity and provenance invariants."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import sequence as seqmod
from flk_repeat.provenance import sha256_of_sequence


def test_accession_and_length():
    rec = seqmod.load_flk_record()
    assert rec["accession"] == "Q9SR13"
    assert rec["length"] == 577
    assert rec["reviewed"] is True


def test_sequence_sha256_matches_provenance_record():
    rec = seqmod.load_flk_record()
    assert rec["sha256"] == rec["provenance"]["sequence_sha256_uppercase"]
    assert sha256_of_sequence(rec["sequence"]) == rec["sha256"]


def test_hash_is_case_insensitive_and_reproducible():
    rec = seqmod.load_flk_record()
    h1 = sha256_of_sequence(rec["sequence"])
    h2 = sha256_of_sequence(rec["sequence"].lower())
    assert h1 == h2

    # reproducible across repeated loads
    rec2 = seqmod.load_flk_record()
    assert rec["sha256"] == rec2["sha256"]


def test_iupred_cache_length_aligned():
    rec = seqmod.load_flk_record()
    scores = seqmod.load_iupred_scores(rec["length"])
    assert len(scores) == rec["length"]
    assert all(0.0 <= s <= 1.0 for s in scores)
