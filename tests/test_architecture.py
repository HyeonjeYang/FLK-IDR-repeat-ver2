"""Repeat-coordinate and composition invariants."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import sequence as seqmod
from flk_repeat import architecture as arch


def test_exactly_three_lepqqyev_copies():
    rec = seqmod.load_flk_record()
    starts = arch.find_motif_occurrences(rec["sequence"])
    assert len(starts) == 3
    for s in starts:
        assert rec["sequence"][s - 1:s - 1 + 8] == "LEPQQYEV"


def test_repeat_coordinates_correct():
    rec = seqmod.load_flk_record()
    starts = arch.find_motif_occurrences(rec["sequence"])
    assert starts == [51, 63, 75]


def test_repeat_spacing():
    starts = [51, 63, 75]
    gaps = arch.repeat_spacing(starts)
    assert gaps == [4, 4]


def test_primary_window_composition_matches_mockler_2004():
    rec = seqmod.load_flk_record()
    start, end = arch.WINDOWS["primary_1_150"]
    comp = arch.window_composition(rec["sequence"], start, end)
    assert comp["length"] == 150
    assert comp["Q_count"] == 33  # Mockler et al. 2004: 33/150 (~22%) Q
    assert abs(comp["Q_fraction"] - 33 / 150) < 1e-9
    assert abs(comp["EQY_fraction"] - (comp["E_count"] + comp["Q_count"] + comp["Y_count"]) / 150) < 1e-9


def test_sensitivity_windows_match_expected_boundaries():
    assert arch.WINDOWS["sensitivity_1_107"] == (1, 107)
    assert arch.WINDOWS["sensitivity_1_182"] == (1, 182)


def test_near_motif_windows_finds_exactly_the_three_repeats():
    rec = seqmod.load_flk_record()
    n150 = rec["sequence"][:150]
    hits = arch.near_motif_windows(n150, max_mismatches=1)
    starts = sorted(h[0] for h in hits)
    assert starts == [51, 63, 75]
    assert all(h[2] == 0 for h in hits)  # all three are exact, zero mismatches
