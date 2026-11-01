r"""Reference-free repeat discovery in each species' own N-terminal region
(revision, Reviewer 1 comment 1: "similarities and differences in the 8-mer
sequences, copy numbers, etc.").

The original analysis asked only whether A. thaliana's LEPQQYEV survives at
the A. thaliana positions. Here every species is scanned on its own terms:
which k-mer is repeated most, how many copies it has (exactly, and allowing
1 or 2 substitutions), and how they are spaced. No alignment and no
reference sequence are involved, so the result cannot be an alignment
artifact.

Statistics
----------
- $T_k$: maximal multiplicity of any exact k-mer (overlapping windows), as
  in `null_model.py`.
- $A_{k,d}$: for every window $s$, the number of windows $w$ with Hamming
  distance $H(w, s) \le d$ (overlapping windows, $s$ itself included);
  $A_{k,d} = \max_s$ of that number. Computed with numpy for whole batches
  of shuffled sequences (`batch_approximate_multiplicity`) so the same
  statistic can be tested against the composition-preserving nulls.
- For reporting, `best_repeat_family` selects the seed with the most
  non-overlapping copies (ties: more overlapping hits, then exact hits,
  then earliest seed), and returns the greedy left-to-right copies and 1-based starts,
  spacing, and the column-wise consensus of the copies.
"""
from collections import Counter

import numpy as np

from .fast_null import encode, _batched_permutations, batch_max_kmer_multiplicity
from .null_model import block_bounds, clopper_pearson_ci


def windows(codes: np.ndarray, k: int) -> np.ndarray:
    """(..., L) -> (..., L-k+1, k) sliding windows (a view)."""
    return np.lib.stride_tricks.sliding_window_view(codes, k, axis=-1)


def batch_approximate_multiplicity(codes: np.ndarray, k: int, d: int) -> np.ndarray:
    """A_{k,d} for every row of a (B, L) code matrix."""
    win = windows(codes, k)  # (B, W, k)
    out = np.empty(codes.shape[0], dtype=np.int32)
    for b in range(codes.shape[0]):
        w = win[b]
        mismatches = (w[:, None, :] != w[None, :, :]).sum(axis=2)
        out[b] = int((mismatches <= d).sum(axis=1).max())
    return out


def best_repeat_family(seq: str, k: int = 8, d: int = 1) -> dict:
    """Best non-overlapping family, with A_{k,d} also reported separately."""
    codes = encode(seq)
    if len(codes) < k:
        return {"k": k, "d": d, "seed": "", "n_overlapping_matches": 0, "copies": [], "starts_1based": []}
    w = windows(codes, k)
    mism = (w[:, None, :] != w[None, :, :]).sum(axis=2)
    approx = (mism <= d).sum(axis=1)
    exact = (mism == 0).sum(axis=1)
    starts_by_seed = []
    for seed_pos in range(len(approx)):
        starts, last_end = [], -1
        for i in np.flatnonzero(mism[seed_pos] <= d):
            if i > last_end:
                starts.append(int(i))
                last_end = i + k - 1
        starts_by_seed.append(starts)
    seed_idx = min(range(len(approx)),
                   key=lambda i: (-len(starts_by_seed[i]), -approx[i], -exact[i], i))
    seed = seq[seed_idx:seed_idx + k]
    starts = starts_by_seed[seed_idx]
    copies = [seq[i:i + k] for i in starts]
    consensus = "".join(Counter(col).most_common(1)[0][0] for col in zip(*copies)) if copies else ""
    return {
        "k": k,
        "d": d,
        "seed": seed,
        "n_overlapping_matches": int(approx.max()),
        "chosen_seed_overlapping_matches": int(approx[seed_idx]),
        "n_copies_nonoverlapping": len(copies),
        "copies": copies,
        "starts_1based": [s + 1 for s in starts],
        "spacing_start_to_start": [b - a for a, b in zip(starts, starts[1:])],
        "consensus": consensus,
    }


def motif_family_copies(seq: str, motif: str, max_mismatches: int) -> list[tuple[int, str, int]]:
    """Non-overlapping windows within `max_mismatches` of `motif` (greedy,
    best-scoring first, then left to right): [(1-based start, window,
    mismatches)]."""
    k = len(motif)
    hits = []
    for i in range(len(seq) - k + 1):
        win = seq[i:i + k]
        mm = sum(a != b for a, b in zip(win, motif))
        if mm <= max_mismatches:
            hits.append((mm, i, win))
    chosen, taken = [], set()
    for mm, i, win in sorted(hits):
        span = set(range(i, i + k))
        if span & taken:
            continue
        taken |= span
        chosen.append((i + 1, win, mm))
    return sorted(chosen)


def repeat_null(seq: str, k: int = 8, d: int = 1, n_permutations: int = 10_000, seed: int = 20261020,
                block_size: int | None = None, phase_offset: int = 0, batch: int = 500) -> dict:
    """Composition-preserving null for both T_k (d = 0 part) and A_{k,d}."""
    codes = encode(seq)
    t_obs = int(batch_max_kmer_multiplicity(codes[None, :], k)[0])
    a_obs = int(batch_approximate_multiplicity(codes[None, :], k, d)[0])
    bounds = None if block_size is None else block_bounds(len(codes), block_size, phase_offset)
    rng = np.random.default_rng(seed)
    n_t, n_a, done = 0, 0, 0
    a_sum = 0
    while done < n_permutations:
        b = min(batch, n_permutations - done)
        perm = _batched_permutations(codes, rng, b, bounds)
        n_t += int(np.sum(batch_max_kmer_multiplicity(perm, k) >= t_obs))
        a = batch_approximate_multiplicity(perm, k, d)
        n_a += int(np.sum(a >= a_obs))
        a_sum += int(a.sum())
        done += b
    t_lo, t_hi = clopper_pearson_ci(n_t, n_permutations)
    a_lo, a_hi = clopper_pearson_ci(n_a, n_permutations)
    return {
        "k": k, "d": d, "n_permutations": n_permutations, "seed": seed,
        "null_model": "global" if block_size is None else "segmental",
        "block_size": block_size,
        "T_observed": t_obs, "T_p_emp": (1 + n_t) / (n_permutations + 1),
        "T_p_ci_lower": t_lo, "T_p_ci_upper": t_hi,
        "A_observed": a_obs, "A_null_mean": a_sum / n_permutations,
        "A_p_emp": (1 + n_a) / (n_permutations + 1),
        "A_p_ci_lower": a_lo, "A_p_ci_upper": a_hi,
    }
