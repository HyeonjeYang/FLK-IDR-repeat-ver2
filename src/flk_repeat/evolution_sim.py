r"""Neutral sequence-evolution simulation for the "are physicochemical
properties conserved beyond sequence identity?" test (revision).

Approach (after Zarin et al. 2019, eLife 8:e46883): evolve a root IDR
sequence down the FLK species/gene tree under an empirical amino-acid
substitution model that knows nothing about the region's function, then
compare the spread of each descriptor across the simulated tips with the
spread across the real orthologs. A descriptor whose real spread is much
smaller than the simulated spread has been held in place more tightly than
the amount of sequence change would predict.

Model
-----
- Exchangeabilities: LG (Le & Gascuel 2008, Mol Biol Evol 25:1307),
  read from PAML's `lg.dat` (data/expanded/models/lg.dat).
- Equilibrium frequencies: amino-acid composition of disordered regions
  of the A. thaliana reference proteome (UniProt UP000006548,
  MobiDB-lite "Disordered" REGION features), so that neutral drift pulls
  sequences towards a *generic* plant IDR composition, not towards FLK's
  own composition.
- Rate matrix $Q_{ij} = s_{ij}\pi_j$ ($i \neq j$), rows summing to zero,
  scaled to one expected substitution per site; transition probabilities
  $P(t) = e^{Qt}$.
- No insertions/deletions are simulated (stated as a limitation), so only
  length-intensive descriptors are tested.
"""
import numpy as np
from scipy.linalg import expm

PAML_ORDER = "ARNDCQEGHILKMFPSTWYV"


def read_paml_exchangeabilities(path) -> np.ndarray:
    """20x20 symmetric exchangeability matrix S from a PAML .dat file
    (lower triangle, 190 numbers, in PAML_ORDER)."""
    numbers = []
    for line in open(path, encoding="utf-8"):
        numbers.extend(float(x) for x in line.split())
        if len(numbers) >= 190:
            break
    s = np.zeros((20, 20))
    idx = 0
    for i in range(1, 20):
        for j in range(i):
            s[i, j] = s[j, i] = numbers[idx]
            idx += 1
    return s


def rate_matrix(exchangeabilities: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    q = exchangeabilities * freqs[None, :]
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    mean_rate = -np.sum(freqs * np.diag(q))
    return q / mean_rate


def composition_vector(sequences, order: str = PAML_ORDER) -> np.ndarray:
    counts = np.zeros(20)
    pos = {aa: i for i, aa in enumerate(order)}
    for s in sequences:
        for c in s:
            if c in pos:
                counts[pos[c]] += 1
    return counts / counts.sum()


def encode(seq: str) -> np.ndarray:
    pos = {aa: i for i, aa in enumerate(PAML_ORDER)}
    return np.array([pos[c] for c in seq if c in pos], dtype=np.int64)


def decode(codes) -> str:
    return "".join(PAML_ORDER[i] for i in codes)


class Simulator:
    """Evolve (replicates x sites) residue arrays along a rooted tree."""

    def __init__(self, q: np.ndarray, rng: np.random.Generator):
        self.q = q
        self.rng = rng
        self._cache = {}

    def transition(self, t: float) -> np.ndarray:
        key = round(float(t), 10)
        if key not in self._cache:
            p = expm(self.q * t)
            p = np.clip(p, 0.0, None)
            self._cache[key] = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
        return self._cache[key]

    def evolve(self, states: np.ndarray, t: float) -> np.ndarray:
        if t <= 0:
            return states.copy()
        cum = self.transition(t)[states]  # (..., 20)
        u = self.rng.random(states.shape + (1,))
        return np.minimum((u > cum).sum(axis=-1), 19)

    def simulate_tree(self, tree: dict, root_states: np.ndarray, n_replicates: int) -> dict:
        """tree: {"name": str|None, "length": float, "children": [...]}.
        Returns {tip name: (n_replicates, n_sites) int array}."""
        tips = {}
        start = np.tile(root_states, (n_replicates, 1))

        def walk(node, states):
            here = self.evolve(states, node.get("length", 0.0))
            children = node.get("children") or []
            if not children:
                tips[node["name"]] = here
            for child in children:
                walk(child, here)

        walk(tree, start)
        return tips


def poisson_corrected_distance(p: float, freqs: np.ndarray) -> float:
    """Tajima-Nei-style correction of a protein p-distance for unequal
    equilibrium frequencies: d = -b ln(1 - p/b), b = 1 - sum(pi^2)."""
    b = 1.0 - float(np.sum(freqs ** 2))
    if p >= b:
        return float("inf")
    return -b * np.log(1.0 - p / b)
