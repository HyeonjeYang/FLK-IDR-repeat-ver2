r"""Gene-family phylogeny used as one of the independent orthology checks
for the expanded FLK candidate set (revision), and the species tree used by
the neutral-evolution simulation.

- Multiple sequence alignment: FAMSA (Deorowicz et al. 2016, Sci Rep
  6:33964) through its Python binding `pyfamsa`, run locally and
  deterministically.
- Distances: protein p-distance over columns where both sequences are
  ungapped, corrected for multiple hits and unequal residue frequencies
  (Tajima & Nei 1984-style correction), capped at `MAX_DISTANCE` when
  saturated. Computed for all pairs at once with matrix products.
- Tree: neighbour joining (Saitou & Nei 1987), implemented with numpy;
  support for every bipartition from `n_bootstrap` column-resampled
  replicates.

Exploratory FLK-side placement (`flk_clade`): the largest set of tips on
one side of a tree edge containing A. thaliana FLK but no listed
A. thaliana paralog (PEP, BTR1, HEN4, RCF3/HOS5, ...). This placement
does not by itself establish one-to-one orthology or even that every tip
is pairwise closer to FLK than to each paralog. Its bootstrap support is
the frequency of the same bipartition among column-resampled replicates.
"""
import numpy as np

MAX_DISTANCE = 4.0
AA = "ACDEFGHIKLMNPQRSTVWY"


def align(records: dict[str, str], threads: int = 0) -> dict[str, str]:
    from pyfamsa import Aligner, Sequence

    aligner = Aligner(guide_tree="upgma", threads=threads)
    msa = aligner.align([Sequence(name.encode(), seq.encode()) for name, seq in records.items()])
    out = {s.id.decode(): s.sequence.decode() for s in msa}
    return {name: out[name] for name in records}


def _one_hot(aln: dict[str, str]) -> tuple[list[str], np.ndarray]:
    names = list(aln)
    arr = np.array([list(aln[n]) for n in names])
    x = np.stack([(arr == a) for a in AA], axis=2).astype(np.float64)  # (n, L, 20)
    return names, x


def distance_matrix(aln: dict[str, str], column_weights: np.ndarray | None = None) -> tuple[list[str], np.ndarray]:
    names, x = _one_hot(aln)
    n, length, _ = x.shape
    w = np.ones(length) if column_weights is None else column_weights
    present = x.sum(axis=2)  # (n, L) 1 where a standard residue
    same = np.einsum("ila,jla,l->ij", x, x, w)
    both = (present * w) @ present.T
    freqs = (x * w[None, :, None]).sum(axis=(0, 1))
    freqs = freqs / freqs.sum()
    b = 1.0 - np.sum(freqs ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(both > 0, 1.0 - same / both, 1.0)
        d = np.where(p < b, -b * np.log(1.0 - p / b), MAX_DISTANCE)
    d = np.minimum(d, MAX_DISTANCE)
    np.fill_diagonal(d, 0.0)
    return names, d


def neighbor_joining(d: np.ndarray) -> list[tuple[int, int, float]]:
    """Edges (node_a, node_b, length) of the unrooted NJ tree. Leaves are
    0..n-1, internal nodes n, n+1, ...; negative branch lengths are set to
    zero."""
    active = list(range(len(d)))
    edges = []
    next_id = len(d)
    mat = d.astype(float).copy()
    while len(active) > 2:
        n = len(active)
        r = mat.sum(axis=1)
        q = (n - 2) * mat - r[:, None] - r[None, :]
        np.fill_diagonal(q, np.inf)
        i, j = np.unravel_index(np.argmin(q), q.shape)
        if i > j:
            i, j = j, i
        li = 0.5 * mat[i, j] + (r[i] - r[j]) / (2 * (n - 2))
        lj = mat[i, j] - li
        u = next_id
        next_id += 1
        edges.append((active[i], u, max(li, 0.0)))
        edges.append((active[j], u, max(lj, 0.0)))
        new_row = 0.5 * (mat[i] + mat[j] - mat[i, j])
        keep = [k for k in range(n) if k not in (i, j)]
        new_mat = np.zeros((n - 1, n - 1))
        new_mat[:-1, :-1] = mat[np.ix_(keep, keep)]
        new_mat[-1, :-1] = new_row[keep]
        new_mat[:-1, -1] = new_row[keep]
        mat = new_mat
        active = [active[k] for k in keep] + [u]
    edges.append((active[0], active[1], max(mat[0, 1], 0.0)))
    return edges


def _adjacency(edges):
    adj = {}
    for a, b, length in edges:
        adj.setdefault(a, []).append((b, length))
        adj.setdefault(b, []).append((a, length))
    return adj


def splits(edges, n_leaves: int) -> list[frozenset]:
    """Leaf set on one side of every internal edge (the side without leaf 0)."""
    adj = _adjacency(edges)
    out = []
    for a, b, _ in edges:
        if a < n_leaves or b < n_leaves:
            continue
        side, stack, seen = set(), [b], {a, b}
        while stack:
            node = stack.pop()
            if node < n_leaves:
                side.add(node)
            for nb, _ in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if 0 in side:
            side = set(range(n_leaves)) - side
        out.append(frozenset(side))
    return out


def bootstrap_nj(aln: dict[str, str], n_bootstrap: int = 200, seed: int = 20261020) -> dict:
    names, d = distance_matrix(aln)
    edges = neighbor_joining(d)
    n_cols = len(next(iter(aln.values())))
    rng = np.random.default_rng(seed)
    counts = {}
    for _ in range(n_bootstrap):
        w = np.bincount(rng.integers(0, n_cols, size=n_cols), minlength=n_cols).astype(float)
        _, db = distance_matrix(aln, w)
        for s in splits(neighbor_joining(db), len(names)):
            counts[s] = counts.get(s, 0) + 1
    support = {s: 100.0 * counts.get(s, 0) / n_bootstrap for s in splits(edges, len(names))}
    return {"names": names, "distances": d, "edges": edges, "support": support, "n_bootstrap": n_bootstrap}


def flk_clade(tree: dict, flk_name: str, paralog_names: list[str]) -> tuple[set[str], float]:
    """Largest bipartition side containing FLK and no listed paralog, with
    its bootstrap support (percent)."""
    names = tree["names"]
    idx = {n: i for i, n in enumerate(names)}
    flk, paralogs = idx[flk_name], {idx[p] for p in paralog_names if p in idx}
    everyone = set(range(len(names)))
    best, best_support = {flk}, float("nan")
    for s, sup in tree["support"].items():
        for side in (set(s), everyone - set(s)):
            if flk in side and not (side & paralogs) and len(side) > len(best):
                best, best_support = side, sup
    return {names[i] for i in best}, best_support


def rooted_tree(edges, names: list[str], outgroup: str) -> dict:
    """Nested {"name", "length", "children"} tree rooted on the branch
    leading to `outgroup` (the root sits at the outgroup's parent node)."""
    adj = _adjacency(edges)
    out_idx = names.index(outgroup)
    root = adj[out_idx][0][0]

    def build(node, parent, length):
        children = [build(nb, node, l) for nb, l in adj[node] if nb != parent]
        if node < len(names):
            return {"name": names[node], "length": length}
        return {"name": None, "length": length, "children": children}

    return build(root, None, 0.0)


def to_newick(tree: dict) -> str:
    def walk(node):
        if not node.get("children"):
            label = node["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        else:
            label = "(" + ",".join(walk(c) for c in node["children"]) + ")"
        return f"{label}:{node.get('length', 0.0):.5f}"
    return walk(tree) + ";"
