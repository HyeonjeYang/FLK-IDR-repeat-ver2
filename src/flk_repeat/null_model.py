r"""Composition-preserving permutation null models for FLK N-terminal
sequence windows (SECONDARY analysis -- see docs/METHODS.md for why the
evolutionary comparison in `homologs.py` is primary).

Two null models are implemented, deliberately kept simple:

- **Global null** (`run_global_permutation_null`): every residue in the
  window is shuffled without replacement. Destroys all order, including
  local low-complexity blocks. Asks: is the repeat multiplicity unusual
  given only the window's overall amino-acid composition?
- **Segmental (local) null** (`run_segmental_permutation_null`): the window
  is cut into fixed, contiguous, non-overlapping blocks (block size chosen
  a priori, before inspecting any result -- 25 aa, see docs/METHODS.md);
  residues are shuffled only within each block. Preserves broad local
  compositional structure while still destroying exact residue order and
  the repeat itself. Asks: is the repeat multiplicity still unusual once
  that coarser local organization is held fixed?

Neither null model is a claim about natural selection, biological function,
or LLPS -- both are strictly descriptive tail tests.

The segmental null's fixed 25-aa block grid has an arbitrary origin: with
repeats starting at residues 51/63/75, the default grid (blocks at
0-25/25-50/.../125-150) places a boundary inside or immediately next to
the third repeat. `run_segmental_permutation_null(..., phase_offset=k)`
shifts where that block grid starts (see `block_bounds`) so
`scripts/run_composition_null.py` can check whether the segmental-null
p-value is an artifact of that one arbitrary boundary choice, at a fixed
set of offsets (0, 5, 10, 15, 20 aa) chosen before looking at any result --
never by scanning for the offset that gives the strongest significance.

Definitions
-----------
An exact 8-mer is counted with an OVERLAPPING sliding window: for a
sequence of length N, window i (1-based start, i = 1..N-7) is seq[i-1:i+7].
`tests/test_null_model.py` checks this definition against a hand-built
example.

    $T_8 = \max_s C(s)$

where $C(s)$ is the number of (overlapping) windows equal to 8-mer $s$.
Each $p_{\mathrm{emp}}$ is reported alongside an exact Clopper-Pearson 95%
CI on the underlying exceedance proportion $k/n$ (`clopper_pearson_ci`).

Analytical motif-specific probability
--------------------------------------
For a specific 8-mer with residue multiplicities $(k_a)_a$ drawn from a
pool of N residues with composition $(n_a)_a$, the probability that one
specific, ordered 8-residue window equals that motif under a uniformly
random permutation of the full N-residue multiset is

    $P(\mathrm{motif}) = \dfrac{\prod_a (n_a)_{k_a}}{(N)_8}$

where $(n)_k = n (n-1) \cdots (n-k+1)$ is the falling factorial. This
follows from placing the 8 required residues into the 8 ordered window
positions one at a time, sampling without replacement from the pool.

    $E[X] = (N-7) \, P(\mathrm{motif})$

is the expected number of (overlapping) windows equal to the motif. Because
overlapping windows are NOT independent (they share residues and are drawn
without replacement from a finite pool), $E[X]$ does NOT directly give
$P(X \geq 3)$; a Poisson approximation using this $E[X]$ as $\lambda$ is
reported only as an analytical sanity check against the permutation
estimate, never as an exact tail probability.
"""
import math
from collections import Counter

import numpy as np
from scipy.stats import beta as _beta_dist
from tqdm import tqdm

from .architecture import MOTIF

DEFAULT_SEED = 20260826  # fixed, documented seed (this project's start date)
DEFAULT_N_PERMUTATIONS = 1_000_000
DEFAULT_BLOCK_SIZE = 25  # chosen a priori for the segmental null; see docs/METHODS.md


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact Clopper-Pearson CI for a binomial proportion k/n (here: the true
    tail probability P(T_null >= T_obs), estimated by k exceedances out of n
    permutations)."""
    alpha = 1 - confidence
    lower = 0.0 if k == 0 else _beta_dist.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else _beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lower), float(upper)


def count_kmers(seq: str, k: int = 8) -> Counter:
    """Overlapping k-mer counts (see module docstring for the definition)."""
    return Counter(seq[i:i + k] for i in range(len(seq) - k + 1))


def max_kmer_multiplicity(seq: str, k: int = 8) -> tuple[int, list[str]]:
    """T_8 (or general T_k) and the k-mer(s) achieving it."""
    counts = count_kmers(seq, k)
    t_max = max(counts.values())
    winners = sorted(s for s, c in counts.items() if c == t_max)
    return t_max, winners


def count_exact_motif(seq: str, motif: str = MOTIF) -> int:
    """Number of overlapping windows exactly equal to `motif`."""
    return count_kmers(seq, len(motif)).get(motif, 0)


def falling_factorial(n: int, k: int) -> int:
    result = 1
    for i in range(k):
        result *= (n - i)
    return result


def poisson_upper_tail(lam: float, k_min: int) -> float:
    r"""Numerically stable $P(X \geq k_{\min})$ for $X \sim \mathrm{Poisson}(\lambda)$.

    Computed by summing the upper-tail PMF terms directly
    ($\sum_{k=k_{\min}}^{\infty} e^{-\lambda}\lambda^k/k!$) rather than as
    $1 - \sum_{k=0}^{k_{\min}-1} P(X=k)$: when $\lambda$ is tiny (as it is
    for the FLK primary window, $\lambda \approx 2.6\times 10^{-6}$) the
    latter subtracts two floating-point numbers that are both within
    $10^{-16}$ of 1.0, so the true tail probability -- itself many orders
    of magnitude smaller than 1 -- is lost entirely to cancellation and the
    naive computation returns exactly 0.0. Direct summation of the (small,
    rapidly-converging) tail terms has no such cancellation."""
    if lam <= 0:
        return 0.0
    term = math.exp(-lam) * lam ** k_min / math.factorial(k_min)
    total = term
    k = k_min
    while term > total * 1e-16:
        k += 1
        term *= lam / k
        total += term
        if k > k_min + 10_000:  # safety cutoff; never reached for realistic lambda
            break
    return total


def analytical_motif_probability(seq: str, motif: str = MOTIF) -> dict:
    """Exact analytical P(one specific 8-window == motif) and E[X] under a
    composition-preserving random permutation of `seq`."""
    n = len(seq)
    composition = Counter(seq)
    motif_multiplicities = Counter(motif)

    numerator = 1
    for aa, k in motif_multiplicities.items():
        numerator *= falling_factorial(composition.get(aa, 0), k)
    denominator = falling_factorial(n, len(motif))

    p_window = numerator / denominator
    n_windows = n - len(motif) + 1
    expected_count = n_windows * p_window

    # Poisson approximation, sanity check only -- see module docstring.
    # poisson_upper_tail is used instead of 1 - P(X<3) to avoid catastrophic
    # cancellation when lambda is tiny (see its docstring).
    lam = expected_count
    poisson_p_ge3 = poisson_upper_tail(lam, 3)

    return {
        "N": n,
        "motif": motif,
        "composition_used": {aa: composition.get(aa, 0) for aa in sorted(set(motif))},
        "P_window_equals_motif": p_window,
        "n_windows": n_windows,
        "E_X_expected_motif_windows": expected_count,
        "poisson_approx_P_X_ge_3": poisson_p_ge3,
    }


def _permutation_summary(seq: str, t8_samples: np.ndarray, motif_count_samples: np.ndarray,
                          seed: int, n_permutations: int, motif: str, extra: dict) -> dict:
    t8_obs, t8_winners = max_kmer_multiplicity(seq, k=len(motif))
    motif_count_obs = count_exact_motif(seq, motif)

    n_ge_t8 = int(np.sum(t8_samples >= t8_obs))
    p_emp_t8 = (1 + n_ge_t8) / (n_permutations + 1)
    ci_lower_t8, ci_upper_t8 = clopper_pearson_ci(n_ge_t8, n_permutations)  # exact 95% CI on k/n
    n_ge3_motif = int(np.sum(motif_count_samples >= 3))
    p_emp_motif_ge3 = (1 + n_ge3_motif) / (n_permutations + 1)

    result = {
        "seed": seed,
        "n_permutations": n_permutations,
        "sequence_length": len(seq),
        "motif": motif,
        "T8_observed": t8_obs,
        "T8_observed_kmers": t8_winners,
        "T8_null_mean": float(np.mean(t8_samples)),
        "T8_null_max": int(np.max(t8_samples)),
        "n_permutations_with_T8_ge_observed": n_ge_t8,
        "p_emp_T8": p_emp_t8,
        "p_emp_T8_ci_lower": ci_lower_t8,
        "p_emp_T8_ci_upper": ci_upper_t8,
        "motif_exact_count_observed": motif_count_obs,
        "n_permutations_with_motif_count_ge_3": n_ge3_motif,
        "p_emp_motif_count_ge_3": p_emp_motif_ge3,
        "t8_samples": t8_samples,
        "motif_count_samples": motif_count_samples,
    }
    result.update(extra)
    return result


def run_global_permutation_null(seq: str, n_permutations: int = DEFAULT_N_PERMUTATIONS,
                                 seed: int = DEFAULT_SEED, motif: str = MOTIF,
                                 progress: bool = False, desc: str = "global null") -> dict:
    """Global composition-preserving null: shuffle every residue of `seq`
    without replacement, `n_permutations` times, fixed documented seed."""
    arr = np.array(list(seq))
    rng = np.random.default_rng(seed)
    k = len(motif)

    t8_samples = np.empty(n_permutations, dtype=np.int32)
    motif_count_samples = np.empty(n_permutations, dtype=np.int32)
    iterator = tqdm(range(n_permutations), desc=desc) if progress else range(n_permutations)
    for i in iterator:
        permuted = "".join(rng.permutation(arr))
        counts = count_kmers(permuted, k=k)
        t8_samples[i] = max(counts.values())
        motif_count_samples[i] = counts.get(motif, 0)

    return _permutation_summary(seq, t8_samples, motif_count_samples, seed, n_permutations,
                                 motif, extra={"null_model": "global", "block_size": None, "phase_offset": None})


def block_bounds(n: int, block_size: int, phase_offset: int = 0) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping block boundaries covering [0, n), with
    the fixed block grid shifted by `phase_offset` residues. phase_offset=0
    is the default grid (blocks start at 0, block_size, 2*block_size, ...);
    a nonzero offset instead starts with one shorter leading block of
    `phase_offset` residues, then full blocks, then a possibly-shorter
    trailing block -- used only to test whether a segmental-null result
    depends on the arbitrary choice of where the block grid starts."""
    phase_offset = phase_offset % block_size
    bounds = []
    if phase_offset > 0:
        bounds.append((0, min(phase_offset, n)))
    start = phase_offset
    while start < n:
        end = min(start + block_size, n)
        bounds.append((start, end))
        start = end
    return bounds


def run_segmental_permutation_null(seq: str, block_size: int = DEFAULT_BLOCK_SIZE,
                                    n_permutations: int = DEFAULT_N_PERMUTATIONS,
                                    seed: int = DEFAULT_SEED, motif: str = MOTIF,
                                    phase_offset: int = 0, progress: bool = False,
                                    desc: str = "segmental null") -> dict:
    """Segmental (local) composition-preserving null: cut `seq` into fixed
    contiguous blocks of `block_size` residues (final block may be shorter),
    optionally starting the block grid `phase_offset` residues in (see
    `block_bounds`), and shuffle residues only within each block, preserving
    per-block composition. Destroys exact residue order and the repeat
    itself while retaining broad local compositional structure."""
    arr = np.array(list(seq))
    n = len(arr)
    bounds = block_bounds(n, block_size, phase_offset)
    rng = np.random.default_rng(seed)
    k = len(motif)

    t8_samples = np.empty(n_permutations, dtype=np.int32)
    motif_count_samples = np.empty(n_permutations, dtype=np.int32)
    out = np.empty(n, dtype=arr.dtype)
    iterator = tqdm(range(n_permutations), desc=desc) if progress else range(n_permutations)
    for i in iterator:
        for lo, hi in bounds:
            out[lo:hi] = rng.permutation(arr[lo:hi])
        permuted = "".join(out)
        counts = count_kmers(permuted, k=k)
        t8_samples[i] = max(counts.values())
        motif_count_samples[i] = counts.get(motif, 0)

    return _permutation_summary(seq, t8_samples, motif_count_samples, seed, n_permutations,
                                 motif, extra={"null_model": "segmental", "block_size": block_size,
                                               "phase_offset": phase_offset})


def histogram_bins(samples: np.ndarray) -> list[dict]:
    """Integer-valued histogram (bin = exact T_8 value) for compact TSV
    storage instead of the full per-permutation sample array."""
    counts = Counter(int(x) for x in samples)
    return [{"T8_value": v, "n_permutations": c} for v, c in sorted(counts.items())]
