r"""Coevolution analyses for the FLK N-terminal repeat region (revision).

Three questions, each answered with its own explicit null:

1. **Do the repeat copies within one species evolve together (concerted
   evolution / homogenization)?** If the three copies were independent,
   ancient "slots", a substitution in one copy would rarely be found at the
   same position, with the same residue, in another copy of the same
   species. `shared_derived_substitutions` counts such within-species
   shared changes relative to the ancestral unit, and
   `concerted_evolution_test` compares that count with copies whose
   substitutions are re-placed at random (same number of substitutions per
   copy; replacement residues drawn from what other species show at that
   unit position).

2. **Do residues co-vary across the protein (intramolecular coevolution)?**
   `mip_matrix` computes average-product-corrected mutual information
   (MIp; Dunn, Wahl & Gloor 2008, Bioinformatics 24:333) between alignment
   columns, with sequence weighting at 80% identity. Its ability to detect
   real coupling in this alignment is checked against residue contacts in
   the AlphaFold model of FLK before any IDR coupling is interpreted.

3. **Do region-level properties change together across species?**
   `phylogenetic_independent_contrasts` (Felsenstein 1985, Am Nat 125:1)
   removes the shared-ancestry dependence between species before two
   traits are correlated.
"""
import itertools
import math

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"


# --------------------------------------------------------------------------
# 1. Concerted evolution of repeat copies
# --------------------------------------------------------------------------

def substitutions(copy: str, ancestral: str) -> set[tuple[int, str]]:
    return {(i, c) for i, (c, a) in enumerate(zip(copy, ancestral)) if c != a}


def shared_derived_substitutions(copies_by_species: dict[str, list[str]], ancestral: str) -> int:
    """Number of (species, position, residue) changes carried by at least
    two copies of the same species."""
    total = 0
    for copies in copies_by_species.values():
        if len(copies) < 2:
            continue
        seen = {}
        for c in copies:
            for sub in substitutions(c, ancestral):
                seen[sub] = seen.get(sub, 0) + 1
        total += sum(1 for n in seen.values() if n >= 2)
    return total


def concerted_evolution_test(copies_by_species: dict[str, list[str]], ancestral: str,
                             n_draws: int = 100_000, seed: int = 20261020) -> dict:
    """Permutation test for `shared_derived_substitutions` (see module
    docstring). Only species with >= 2 copies contribute to the statistic,
    but every species contributes to the per-position replacement pools."""
    k = len(ancestral)
    observed = shared_derived_substitutions(copies_by_species, ancestral)
    pools = {sp: [[] for _ in range(k)] for sp in copies_by_species}
    for sp in copies_by_species:
        for other, copies in copies_by_species.items():
            if other == sp:
                continue
            for c in copies:
                for pos, res in substitutions(c, ancestral):
                    pools[sp][pos].append(res)
    rng = np.random.default_rng(seed)
    focal = {sp: cs for sp, cs in copies_by_species.items() if len(cs) >= 2}
    counts = np.zeros(n_draws, dtype=np.int32)
    for d in range(n_draws):
        total = 0
        for sp, copies in focal.items():
            seen = {}
            for c in copies:
                m = len(substitutions(c, ancestral))
                for pos in rng.choice(k, size=m, replace=False):
                    pool = pools[sp][pos] or [a for a in AA20 if a != ancestral[pos]]
                    res = pool[rng.integers(len(pool))]
                    seen[(pos, res)] = seen.get((pos, res), 0) + 1
            total += sum(1 for n in seen.values() if n >= 2)
        counts[d] = total
    n_ge = int(np.sum(counts >= observed))
    return {
        "observed_shared_derived": observed,
        "null_mean": float(counts.mean()),
        "n_draws": n_draws,
        "n_null_ge_observed": n_ge,
        "p_emp": (1 + n_ge) / (n_draws + 1),
        "n_species_with_multiple_copies": len(focal),
    }


def within_vs_between_species_distance(copies_by_species: dict[str, list[str]]) -> dict:
    """Mean Hamming distance between copies of the same species vs copies
    of different species (descriptive companion to the test above)."""
    labelled = [(sp, c) for sp, cs in copies_by_species.items() for c in cs]
    within, between = [], []
    for (s1, c1), (s2, c2) in itertools.combinations(labelled, 2):
        d = sum(a != b for a, b in zip(c1, c2))
        (within if s1 == s2 else between).append(d)
    return {
        "mean_within_species": float(np.mean(within)) if within else float("nan"),
        "mean_between_species": float(np.mean(between)) if between else float("nan"),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
    }


# --------------------------------------------------------------------------
# 2. Residue covariation (MIp)
# --------------------------------------------------------------------------

def sequence_weights(msa: list[str], identity: float = 0.8) -> np.ndarray:
    """w_s = 1 / #(sequences with >= `identity` fractional identity to s,
    over columns where both are ungapped), including s itself."""
    arr = np.array([list(s) for s in msa])
    n = len(msa)
    weights = np.zeros(n)
    gap = arr == "-"
    for i in range(n):
        both = ~gap[i] & ~gap
        same = (arr == arr[i]) & both
        frac = same.sum(axis=1) / np.maximum(both.sum(axis=1), 1)
        weights[i] = 1.0 / np.sum(frac >= identity)
    return weights


def _encode_msa(msa: list[str]) -> np.ndarray:
    alphabet = {a: i for i, a in enumerate(AA20 + "-")}
    return np.array([[alphabet.get(c, 20) for c in s] for s in msa], dtype=np.int64)


def mip_matrix(msa: list[str], columns: list[int], weights: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    """MIp for the given alignment columns (gaps treated as a 21st state).
    Returns a (len(columns), len(columns)) symmetric matrix, diagonal 0."""
    x = _encode_msa(msa)[:, columns]
    q = 21
    w = weights / weights.sum()
    n_eff_pc = pseudocount / weights.sum()
    m = len(columns)
    fi = np.zeros((m, q))
    for a in range(q):
        fi[:, a] = ((x == a) * w[:, None]).sum(axis=0)
    fi = (1 - n_eff_pc) * fi + n_eff_pc / q
    mi = np.zeros((m, m))
    onehot = np.stack([(x == a).astype(float) for a in range(q)], axis=2)  # (n, m, q)
    weighted = onehot * w[:, None, None]
    for i in range(m):
        fij = np.einsum("na,nmb->mab", weighted[:, i, :], onehot)  # (m, q, q)
        fij = (1 - n_eff_pc) * fij + n_eff_pc / (q * q)
        ratio = fij / (fi[i][None, :, None] * fi[:, None, :])
        mi[i] = np.sum(fij * np.log(ratio), axis=(1, 2))
    np.fill_diagonal(mi, 0.0)
    mean_row = mi.sum(axis=1) / (m - 1)
    mean_all = mi.sum() / (m * (m - 1))
    apc = np.outer(mean_row, mean_row) / mean_all
    out = mi - apc
    np.fill_diagonal(out, 0.0)
    return out


def column_shuffle(msa: list[str], rng: np.random.Generator) -> list[str]:
    """Shuffle each column independently across sequences: keeps every
    column's conservation, destroys any covariation between columns."""
    arr = np.array([list(s) for s in msa])
    for j in range(arr.shape[1]):
        arr[:, j] = arr[rng.permutation(arr.shape[0]), j]
    return ["".join(row) for row in arr]


# --------------------------------------------------------------------------
# 3. Phylogenetic independent contrasts
# --------------------------------------------------------------------------

def phylogenetic_independent_contrasts(tree: dict, traits: dict[str, float]) -> list[float]:
    """Standardized contrasts for one trait on a rooted tree given as
    {"name", "length", "children"}; polytomies are resolved left to right
    with zero-length internal branches. Tips missing from `traits` are
    pruned."""
    contrasts = []

    def walk(node):
        children = node.get("children") or []
        if not children:
            if node["name"] in traits and not math.isnan(traits[node["name"]]):
                return traits[node["name"]], node.get("length", 0.0)
            return None
        results = [r for r in (walk(c) for c in children) if r is not None]
        if not results:
            return None
        value, length = results[0]
        for v2, l2 in results[1:]:
            total = length + l2
            if total <= 0:
                total = 1e-9
            contrasts.append((value - v2) / math.sqrt(total))
            value = (value / length + v2 / l2) / (1 / length + 1 / l2) if length > 0 and l2 > 0 else (value + v2) / 2
            length = length * l2 / total
        return value, length + node.get("length", 0.0)

    walk(tree)
    return contrasts


def contrast_correlation(tree: dict, x: dict[str, float], y: dict[str, float]) -> dict:
    """Correlation through the origin of PICs for two traits measured on
    the same set of tips (Garland, Harvey & Ives 1992)."""
    shared = {k for k in x if k in y and not math.isnan(x[k]) and not math.isnan(y[k])}
    cx = np.array(phylogenetic_independent_contrasts(tree, {k: x[k] for k in shared}))
    cy = np.array(phylogenetic_independent_contrasts(tree, {k: y[k] for k in shared}))
    n = len(cx)
    r = float(np.sum(cx * cy) / math.sqrt(np.sum(cx ** 2) * np.sum(cy ** 2)))
    from scipy.stats import t as t_dist
    df = n - 1
    t_stat = r * math.sqrt(df / max(1e-12, 1 - r ** 2))
    p = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    return {"n_contrasts": n, "r_through_origin": r, "t": t_stat, "df": df, "p_two_sided": float(p)}
