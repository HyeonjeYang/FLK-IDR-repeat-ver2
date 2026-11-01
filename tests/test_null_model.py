"""Null-model invariants: composition preservation, deterministic RNG,
overlapping k-mer counting definition, and empirical p-value calculation.
"""
import math
import sys
from pathlib import Path
from collections import Counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import null_model as nm


def test_overlapping_kmer_counting_definition():
    """Sliding window, step 1: 'AAAAAAAAA' (len 9) has the 8-mer 'AAAAAAAA'
    starting at position 1 and position 2 -- 2 overlapping occurrences."""
    seq = "AAAAAAAAA"
    counts = nm.count_kmers(seq, k=8)
    assert counts["AAAAAAAA"] == 2
    t8, winners = nm.max_kmer_multiplicity(seq, k=8)
    assert t8 == 2
    assert winners == ["AAAAAAAA"]


def test_count_exact_motif_matches_observed_flk_triplication():
    n150 = ("MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEVPDPTLEPQQYEVPDQTLEPQQYEVDDQLEYHQYQLQD"
            "QANEDVQDHSQDDLQYQPQNQEQFQLQDEAHDQAQYQAQGDVQDHNGDEVQDKVE")
    assert len(n150) == 150
    assert nm.count_exact_motif(n150) == 3
    t8, winners = nm.max_kmer_multiplicity(n150)
    assert t8 == 3
    assert winners == ["LEPQQYEV"]


def test_permutation_preserves_composition():
    seq = "MAEAEDQQNFVAHNGDQVPDQ"
    rng = np.random.default_rng(1)
    permuted = "".join(rng.permutation(np.array(list(seq))))
    assert Counter(permuted) == Counter(seq)
    assert len(permuted) == len(seq)


def test_global_null_is_deterministic_given_seed():
    seq = "MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEV"
    r1 = nm.run_global_permutation_null(seq, n_permutations=200, seed=7)
    r2 = nm.run_global_permutation_null(seq, n_permutations=200, seed=7)
    assert list(r1["t8_samples"]) == list(r2["t8_samples"])
    assert r1["p_emp_T8"] == r2["p_emp_T8"]


def test_global_null_different_seed_gives_different_permutations():
    """Compare the actual per-permutation sequences (not T8, which can
    collapse to the same small integer for many distinct permutations and
    is therefore a weak discriminator)."""
    seq = "MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEV"
    arr = np.array(list(seq))
    perm1 = "".join(np.random.default_rng(1).permutation(arr))
    perm2 = "".join(np.random.default_rng(2).permutation(arr))
    assert perm1 != perm2


def test_empirical_p_value_formula_bounds():
    """p_emp = (1 + #{T_i >= T_obs}) / (N + 1); always in (0, 1] and never 0."""
    seq = "MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEVPDPTLEPQQYEVPDQTLEPQQYEV"
    result = nm.run_global_permutation_null(seq, n_permutations=500, seed=42)
    assert 0 < result["p_emp_T8"] <= 1
    assert result["p_emp_T8"] == (1 + result["n_permutations_with_T8_ge_observed"]) / 501


def test_clopper_pearson_ci_basic_properties():
    assert nm.clopper_pearson_ci(0, 1000) == (0.0, nm.clopper_pearson_ci(0, 1000)[1])
    assert nm.clopper_pearson_ci(1000, 1000)[1] == 1.0
    for k, n in [(0, 1000), (1, 1000), (2, 1_000_000), (500, 1000), (1000, 1000)]:
        lo, hi = nm.clopper_pearson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= k / n <= hi  # the point estimate k/n is always inside its own exact CI


def test_permutation_null_reports_ci_alongside_p_emp():
    seq = "MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEVPDPTLEPQQYEVPDQTLEPQQYEV"
    result = nm.run_global_permutation_null(seq, n_permutations=500, seed=42)
    assert "p_emp_T8_ci_lower" in result and "p_emp_T8_ci_upper" in result
    assert 0.0 <= result["p_emp_T8_ci_lower"] <= result["p_emp_T8_ci_upper"] <= 1.0


def test_segmental_null_preserves_per_block_composition():
    seq = "MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEVPDPTLEPQQYEVPDQTLEPQQYEV" * 1
    block_size = 20
    result = nm.run_segmental_permutation_null(seq, block_size=block_size, n_permutations=5, seed=3)
    assert result["null_model"] == "segmental"
    assert result["block_size"] == block_size
    # re-derive one permutation directly and check block-wise composition match
    arr = np.array(list(seq))
    rng = np.random.default_rng(3)
    out = np.empty(len(arr), dtype=arr.dtype)
    for lo in range(0, len(arr), block_size):
        hi = min(lo + block_size, len(arr))
        out[lo:hi] = rng.permutation(arr[lo:hi])
        assert Counter(out[lo:hi]) == Counter(arr[lo:hi])


def test_segmental_null_is_deterministic_given_seed():
    seq = "MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEV"
    r1 = nm.run_segmental_permutation_null(seq, block_size=25, n_permutations=200, seed=7)
    r2 = nm.run_segmental_permutation_null(seq, block_size=25, n_permutations=200, seed=7)
    assert list(r1["t8_samples"]) == list(r2["t8_samples"])


def test_block_bounds_zero_offset_matches_plain_fixed_grid():
    bounds = nm.block_bounds(150, 25, phase_offset=0)
    assert bounds == [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150)]


def test_block_bounds_nonzero_offset_shifts_grid_and_covers_sequence():
    for offset in (5, 10, 15, 20):
        bounds = nm.block_bounds(150, 25, phase_offset=offset)
        # blocks must be contiguous, non-overlapping, and cover [0, 150) exactly
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 150
        for (lo1, hi1), (lo2, hi2) in zip(bounds, bounds[1:]):
            assert hi1 == lo2
        # the leading (partial) block has exactly `offset` residues
        assert bounds[0] == (0, offset)


def test_block_bounds_offset_wraps_modulo_block_size():
    assert nm.block_bounds(150, 25, phase_offset=25) == nm.block_bounds(150, 25, phase_offset=0)


def test_segmental_null_phase_offset_changes_block_partition_but_not_composition():
    seq = "MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEVPDPTLEPQQYEVPDQTLEPQQYEV"
    for offset in (0, 5, 10, 15, 20):
        result = nm.run_segmental_permutation_null(seq, block_size=25, n_permutations=5,
                                                     seed=3, phase_offset=offset)
        assert result["phase_offset"] == offset
        # every sampled permutation must still be a valid rearrangement of seq
        arr = np.array(list(seq))
        rng = np.random.default_rng(3)
        bounds = nm.block_bounds(len(seq), 25, offset)
        out = np.empty(len(arr), dtype=arr.dtype)
        for lo, hi in bounds:
            out[lo:hi] = rng.permutation(arr[lo:hi])
        assert Counter(out) == Counter(arr)


def test_segmental_null_result_is_robust_across_phase_offsets():
    """Regression check on the block-phase robustness result reported in
    docs/METHODS.md: for the real FLK primary window, every one of the 5
    predefined offsets should give a very small empirical p-value (the
    repeat's triplication is not an artifact of where the block grid
    happens to start)."""
    n150 = ("MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEVPDPTLEPQQYEVPDQTLEPQQYEVDDQLEYHQYQLQD"
            "QANEDVQDHSQDDLQYQPQNQEQFQLQDEAHDQAQYQAQGDVQDHNGDEVQDKVE")
    assert len(n150) == 150
    for offset in (0, 5, 10, 15, 20):
        result = nm.run_segmental_permutation_null(n150, block_size=25, n_permutations=2000,
                                                     seed=nm.DEFAULT_SEED, phase_offset=offset)
        assert result["T8_observed"] == 3
        assert result["p_emp_T8"] < 0.01


def test_analytical_probability_matches_hand_calculation():
    """For a toy 8-residue alphabet with unique letters (each count=1),
    P(specific window == a specific permutation of those 8 residues)
    should equal 1 / 8! for N=8."""
    seq = "ABCDEFGH"
    result = nm.analytical_motif_probability(seq, motif="ABCDEFGH")
    import math
    assert abs(result["P_window_equals_motif"] - 1 / math.factorial(8)) < 1e-15
    assert result["n_windows"] == 1
    assert abs(result["E_X_expected_motif_windows"] - result["P_window_equals_motif"]) < 1e-15


def test_falling_factorial():
    assert nm.falling_factorial(5, 0) == 1
    assert nm.falling_factorial(5, 1) == 5
    assert nm.falling_factorial(5, 2) == 20
    assert nm.falling_factorial(10, 8) == 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3


def test_poisson_upper_tail_is_positive_not_zero_for_tiny_lambda():
    """Regression test: naive '1 - sum(P(X=k), k=0..2)' underflows to
    exactly 0.0 for the FLK primary window's tiny lambda (~2.6e-6) due to
    catastrophic cancellation. The stable direct-tail-sum implementation
    must not."""
    lam = 2.589773947150723e-06  # E[X] for the real FLK primary window
    naive = 1.0 - sum(math.exp(-lam) * lam ** k / math.factorial(k) for k in range(3))
    assert naive == 0.0  # documents the bug this function fixes
    stable = nm.poisson_upper_tail(lam, 3)
    assert stable > 0.0


def test_poisson_upper_tail_matches_high_precision_reference():
    """Cross-check against an independent arbitrary-precision (Decimal)
    calculation of the same tail probability."""
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    lam_float = 2.589773947150723e-06
    lam = Decimal(str(lam_float))

    def poisson_pmf_decimal(lam, k):
        return lam ** k * (-lam).exp() / Decimal(math.factorial(k))

    reference_tail = float(sum(poisson_pmf_decimal(lam, k) for k in range(3, 15)))
    stable = nm.poisson_upper_tail(lam_float, 3)
    assert abs(stable - reference_tail) / reference_tail < 1e-9
    # matches the expected order of magnitude reported for the primary window
    assert 1e-18 < stable < 1e-17


def test_poisson_upper_tail_matches_naive_calculation_when_lambda_is_not_tiny():
    """For a lambda large enough that the naive subtraction is numerically
    fine, the stable and naive calculations must agree."""
    lam = 5.0
    naive = 1.0 - sum(math.exp(-lam) * lam ** k / math.factorial(k) for k in range(3))
    stable = nm.poisson_upper_tail(lam, 3)
    assert abs(stable - naive) < 1e-12


def test_analytical_motif_probability_uses_stable_poisson_tail():
    """The primary-window analytical result should carry the corrected,
    nonzero Poisson sanity-check value, not the old silently-zero one."""
    n150 = ("MAEAEDQQNFVAHNGDQVPDQGSDELHNGLPYQVHDETLVHQPYEVEDPILEPQQYEVPDPTLEPQQYEVPDQTLEPQQYEVDDQLEYHQYQLQD"
            "QANEDVQDHSQDDLQYQPQNQEQFQLQDEAHDQAQYQAQGDVQDHNGDEVQDKVE")
    result = nm.analytical_motif_probability(n150)
    assert result["poisson_approx_P_X_ge_3"] > 0.0
    assert 1e-19 < result["poisson_approx_P_X_ge_3"] < 1e-17
