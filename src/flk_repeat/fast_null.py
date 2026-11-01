r"""Vectorized (numpy, batched) versions of the composition-preserving
permutation nulls in `null_model.py`, used for the per-species,
reference-free repeat screen of the expanded homolog set (revision).

`null_model.py` remains the implementation behind every A. thaliana result
reported in the original manuscript; this module exists only because the
pure-Python loop there is too slow to run on ~60 species. The statistic is
identical -- $T_k = \max_s C(s)$ over overlapping exact k-mers -- and
`tests/test_fast_null.py` checks, on fixed and random sequences, that
`batch_max_kmer_multiplicity` returns exactly what
`null_model.max_kmer_multiplicity` returns. The permutation *streams*
differ (rows are shuffled with `Generator.permuted` instead of one
`Generator.permutation` call per draw), so p-values agree in distribution,
not draw-for-draw.
"""
import numpy as np

from .null_model import block_bounds, clopper_pearson_ci

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}
_BASE = len(ALPHABET)


def encode(seq: str) -> np.ndarray:
    return np.array([_INDEX.get(c, _INDEX["X"]) for c in seq.upper()], dtype=np.int64)


def _window_hashes(codes: np.ndarray, k: int) -> np.ndarray:
    """codes: (B, L) int array -> (B, L-k+1) int64 hash of every
    overlapping k-mer (base-21 positional encoding, collision-free for
    k <= 14)."""
    if k > 14:
        raise ValueError("k > 14 would overflow the int64 k-mer hash")
    n_win = codes.shape[1] - k + 1
    h = np.zeros((codes.shape[0], n_win), dtype=np.int64)
    for m in range(k):
        h = h * _BASE + codes[:, m:m + n_win]
    return h


def batch_max_kmer_multiplicity(codes: np.ndarray, k: int = 8) -> np.ndarray:
    """T_k for every row of a (B, L) code matrix."""
    s = np.sort(_window_hashes(codes, k), axis=1)
    eq = s[:, 1:] == s[:, :-1]
    run = np.zeros(s.shape[0], dtype=np.int32)
    best = np.zeros(s.shape[0], dtype=np.int32)
    for j in range(eq.shape[1]):
        run = np.where(eq[:, j], run + 1, 0)
        np.maximum(best, run, out=best)
    return best + 1


def _batched_permutations(codes: np.ndarray, rng: np.random.Generator, batch: int,
                          bounds: list[tuple[int, int]] | None) -> np.ndarray:
    tiled = np.tile(codes, (batch, 1))
    if bounds is None:
        return rng.permuted(tiled, axis=1)
    out = np.empty_like(tiled)
    for lo, hi in bounds:
        out[:, lo:hi] = rng.permuted(tiled[:, lo:hi], axis=1)
    return out


def permutation_null(seq: str, k: int = 8, n_permutations: int = 100_000, seed: int = 20260826,
                     block_size: int | None = None, phase_offset: int = 0,
                     batch: int = 20_000) -> dict:
    """Global null (block_size=None) or segmental null (shuffle within
    contiguous blocks, see `null_model.block_bounds`). Returns the observed
    T_k, the number of permutations reaching it, the +1-corrected empirical
    p-value, and its exact Clopper-Pearson 95% CI."""
    codes = encode(seq)
    t_obs = int(batch_max_kmer_multiplicity(codes[None, :], k)[0])
    bounds = None if block_size is None else block_bounds(len(codes), block_size, phase_offset)
    rng = np.random.default_rng(seed)
    n_ge, done, null_sum = 0, 0, 0
    while done < n_permutations:
        b = min(batch, n_permutations - done)
        t = batch_max_kmer_multiplicity(_batched_permutations(codes, rng, b, bounds), k)
        n_ge += int(np.sum(t >= t_obs))
        null_sum += int(t.sum())
        done += b
    lo, hi = clopper_pearson_ci(n_ge, n_permutations)
    return {
        "k": k,
        "T_observed": t_obs,
        "n_permutations": n_permutations,
        "seed": seed,
        "null_model": "global" if block_size is None else "segmental",
        "block_size": block_size,
        "phase_offset": phase_offset if block_size is not None else None,
        "n_permutations_ge_observed": n_ge,
        "null_mean_T": null_sum / n_permutations,
        "p_emp": (1 + n_ge) / (n_permutations + 1),
        "p_emp_ci_lower": lo,
        "p_emp_ci_upper": hi,
    }
