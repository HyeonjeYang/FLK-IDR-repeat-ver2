"""Neighbour joining, bipartitions and the FLK-clade call on hand-checkable
examples."""
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import phylo as ph

# Additive five-taxon example (Saitou & Nei 1987 algorithm; the textbook
# tree is a:2, b:3 joined to an internal node, then c:4, d:2, e:1).
D5 = np.array([[0, 5, 9, 9, 8],
               [5, 0, 10, 10, 9],
               [9, 10, 0, 8, 7],
               [9, 10, 8, 0, 3],
               [8, 9, 7, 3, 0]], dtype=float)


def _leaf_lengths(edges, n):
    return {min(a, b): l for a, b, l in edges if min(a, b) < n}


def test_nj_recovers_additive_tree():
    edges = ph.neighbor_joining(D5)
    lengths = _leaf_lengths(edges, 5)
    assert math.isclose(lengths[0], 2.0) and math.isclose(lengths[1], 3.0)
    assert math.isclose(lengths[2], 4.0) and math.isclose(lengths[3], 2.0) and math.isclose(lengths[4], 1.0)
    assert math.isclose(sum(l for _, _, l in edges), 2 + 3 + 4 + 2 + 1 + 3 + 2)
    s = {frozenset(x) for x in ph.splits(edges, 5)}
    assert frozenset({2, 3, 4}) in s and frozenset({3, 4}) in s


def test_flk_clade_excludes_paralogs():
    aln = {
        "FLK_At": "MKVLEEQQRKAAG",
        "FLK_sp1": "MKVLEEQQRKAAA",
        "FLK_sp2": "MKVLDEQQRKSAA",
        "PEP_At": "MRILKDNNHTWGG",
        "PEP_sp1": "MRILKDNNHTWGA",
    }
    tree = ph.bootstrap_nj(aln, n_bootstrap=20, seed=1)
    members, support = ph.flk_clade(tree, "FLK_At", ["PEP_At"])
    assert members == {"FLK_At", "FLK_sp1", "FLK_sp2"}
    assert support > 50


def test_distance_matrix_identity_and_symmetry():
    aln = {"a": "ACDEFG", "b": "ACDEFG", "c": "ACDKKK"}
    names, d = ph.distance_matrix(aln)
    assert d[0, 1] == 0.0 and np.allclose(d, d.T) and d[0, 2] > 0


def test_rooted_tree_and_newick():
    edges = ph.neighbor_joining(D5)
    tree = ph.rooted_tree(edges, ["a", "b", "c", "d", "e"], "a")
    nwk = ph.to_newick(tree)
    assert nwk.endswith(";") and all(x in nwk for x in "abcde")
