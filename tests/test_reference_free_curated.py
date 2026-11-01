"""Checks the distinction between overlapping A8,d and reported copy counts."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from flk_repeat.repeats import best_repeat_family
from analyze_curated_reference_free import analyze


def test_approximate_family_counts_cannot_decrease_with_tolerance():
    table = analyze()
    assert len(table) == 22
    assert (table.best_8mer_2_mismatch_nonoverlapping_copies >=
            table.best_8mer_1_mismatch_nonoverlapping_copies).all()
    ath = table.loc[table.accession == "Q9SR13"].iloc[0]
    assert ath.T8_exact_overlapping == 3
    assert ath.best_8mer_1_mismatch_seed == "LEPQQYEV"


def test_overlap_statistic_remains_maximum_when_seed_is_selected_by_copies():
    seq = "AAAAAAAAAA" + "LEPQQYEV" * 3
    fam = best_repeat_family(seq, 8, 1)
    assert fam["n_overlapping_matches"] >= fam["chosen_seed_overlapping_matches"]
    assert fam["n_copies_nonoverlapping"] >= 3
