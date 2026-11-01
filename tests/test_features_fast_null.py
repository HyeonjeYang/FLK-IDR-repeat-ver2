"""Physicochemical descriptors and the vectorized permutation null:
reference values, invariance of composition descriptors under shuffling,
and exact agreement of the fast T_k computation with null_model.py.
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import fast_null as fn
from flk_repeat import features as ft
from flk_repeat import null_model as nm

FLK_1_150 = ("MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEVPDPTLEPQQYEVPDQTLEPQQYEVDDQLEYHQYQLQD"
             "QANEDVQDHSQDDLQYQPQNQEQFQLQDEAHDQAQYQAQGDVQDHNGDEVQDKVE")


def test_patterning_z_separates_blocky_from_evenly_spaced():
    """A class packed into one block is far more clustered than shuffles
    (z >> 0); a class placed every 4th residue is more evenly spaced than
    shuffles (z < 0); absent classes are undefined."""
    blocky = "E" * 12 + "G" * 48
    even = "EGGG" * 15
    assert ft.patterning_z(blocky, {"E"}) > 5
    assert ft.patterning_z(even, {"E"}) < -1
    assert math.isnan(ft.patterning_z("GGGGGGGGGG", {"E"}))


def test_patterning_z_is_deterministic():
    a = ft.patterning_z(FLK_1_150, ft.PATTERNING_CLASSES["acidic"])
    b = ft.patterning_z(FLK_1_150, ft.PATTERNING_CLASSES["acidic"])
    assert a == b


def test_scd_orders_mixed_vs_segregated():
    assert ft.scd("E" * 25 + "K" * 25) < ft.scd("EK" * 25)
    assert ft.scd("QQQQQQ") == 0.0


def test_hand_computed_composition_descriptors():
    f = ft.compute_features("DEKRQQ")
    assert math.isclose(f["fcr"], 4 / 6)
    assert math.isclose(f["ncpr"], 0.0)
    assert math.isclose(f["fraction_polar"], 2 / 6)
    assert math.isclose(f["hydropathy_kd"], (-3.5 - 3.5 - 3.9 - 4.5 - 3.5 - 3.5) / 6)
    assert math.isclose(f["hydrophilicity_hw"], (3.0 + 3.0 + 3.0 + 3.0 + 0.2 + 0.2) / 6)


def test_composition_descriptors_invariant_under_shuffle():
    rng = np.random.default_rng(7)
    shuffled = "".join(rng.permutation(list(FLK_1_150)))
    a, b = ft.compute_features(FLK_1_150), ft.compute_features(shuffled)
    for name in ft.COMPOSITION_FEATURES:
        assert math.isclose(a[name], b[name], rel_tol=1e-9), name


def test_flk_window_t8_feature():
    assert ft.compute_features(FLK_1_150)["t8_max_exact_8mer_multiplicity"] == 3


def test_fast_tk_matches_reference_implementation():
    rng = np.random.default_rng(11)
    seqs = [FLK_1_150, "AAAAAAAAA", "LEPQQYEV" * 4]
    seqs += ["".join(rng.choice(list("QQQEEYLPV"), size=120)) for _ in range(200)]
    for k in (6, 8, 10):
        for s in (s for s in seqs if len(s) >= k):
            expected = nm.max_kmer_multiplicity(s, k=k)[0]
            got = int(fn.batch_max_kmer_multiplicity(fn.encode(s)[None, :], k)[0])
            assert got == expected, (k, s)


def test_fast_permutations_preserve_composition_and_blocks():
    codes = fn.encode(FLK_1_150)
    rng = np.random.default_rng(3)
    glob = fn._batched_permutations(codes, rng, 50, None)
    assert all(sorted(row) == sorted(codes) for row in glob)
    bounds = nm.block_bounds(len(codes), 25, 10)
    seg = fn._batched_permutations(codes, rng, 50, bounds)
    for row in seg:
        for lo, hi in bounds:
            assert sorted(row[lo:hi]) == sorted(codes[lo:hi])


def test_fast_null_reproduces_flk_global_tail_in_distribution():
    """The original 1,000,000-permutation global null on FLK 1-150 gave
    k = 14 exceedances (p_emp = 1.5e-5). With an independent stream, the
    fast null's exceedance rate over 200,000 draws must be compatible."""
    res = fn.permutation_null(FLK_1_150, n_permutations=200_000, seed=99)
    assert res["T_observed"] == 3
    lo, hi = nm.clopper_pearson_ci(res["n_permutations_ge_observed"], res["n_permutations"], 0.999)
    assert lo <= 14 / 1_000_000 <= hi
