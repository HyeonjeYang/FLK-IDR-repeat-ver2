"""Coevolution statistics and the neutral-evolution simulator: hand-worked
examples and model invariants."""
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import coevolution as co
from flk_repeat import evolution_sim as es

LG_PATH = REPO_ROOT / "data" / "expanded" / "models" / "lg.dat"


def test_shared_derived_substitutions_counts_within_species_only():
    copies = {
        "Arabis": ["LEAQQYEV", "LEAQQYGV", "LEAQQFGV"],  # P->A x3, E->G x2, Y->F x1
        "Other": ["LEAQQYEV"],                            # single copy: never counted
    }
    assert co.shared_derived_substitutions(copies, "LEPQQYEV") == 2


def test_concerted_test_detects_homogenized_copies():
    copies = {
        "sp1": ["LEAQQYEV"] * 3,
        "sp2": ["LESQQYEV"] * 2,
        "sp3": ["LEPQEYEV", "LEPQQYEV"],
        "sp4": ["LEPQQYEL", "LEPQQYEV", "LEPQQYEV"],
    }
    res = co.concerted_evolution_test(copies, "LEPQQYEV", n_draws=2000, seed=1)
    assert res["observed_shared_derived"] == 2
    assert res["p_emp"] < 0.05


def test_pic_hand_worked_example():
    tree = {"name": None, "length": 0.0, "children": [
        {"name": None, "length": 1.0, "children": [
            {"name": "A", "length": 1.0}, {"name": "B", "length": 1.0}]},
        {"name": "C", "length": 2.0}]}
    c = co.phylogenetic_independent_contrasts(tree, {"A": 1.0, "B": 3.0, "C": 6.0})
    assert math.isclose(c[0], (1 - 3) / math.sqrt(2))
    assert math.isclose(c[1], (2 - 6) / math.sqrt(1.5 + 2.0))


def test_mip_ranks_covarying_pair_first():
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(300):
        a = rng.choice(list("DEKR"))
        partner = {"D": "K", "E": "R", "K": "D", "R": "E"}[a]
        noise = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=4))
        seqs.append(a + noise + partner)
    w = co.sequence_weights(seqs, 0.8)
    m = co.mip_matrix(seqs, list(range(6)), w)
    i, j = np.unravel_index(np.argmax(m), m.shape)
    assert {i, j} == {0, 5}


def test_lg_rate_matrix_is_normalized_and_converges_to_equilibrium():
    s = es.read_paml_exchangeabilities(LG_PATH)
    assert np.allclose(s, s.T)
    freqs = np.full(20, 0.05)
    q = es.rate_matrix(s, freqs)
    assert np.allclose(q.sum(axis=1), 0.0)
    assert math.isclose(-np.sum(freqs * np.diag(q)), 1.0)
    sim = es.Simulator(q, np.random.default_rng(0))
    cum = sim.transition(50.0)
    p = np.diff(np.concatenate([np.zeros((20, 1)), cum], axis=1), axis=1)
    assert np.allclose(p, freqs[None, :], atol=1e-6)


def test_simulation_preserves_length_and_tip_set():
    s = es.read_paml_exchangeabilities(LG_PATH)
    q = es.rate_matrix(s, np.full(20, 0.05))
    sim = es.Simulator(q, np.random.default_rng(1))
    tree = {"name": None, "length": 0.0, "children": [
        {"name": "A", "length": 0.1}, {"name": "B", "length": 0.3}]}
    root = es.encode("LEPQQYEVPDPTLEPQQYEV")
    tips = sim.simulate_tree(tree, root, n_replicates=50)
    assert set(tips) == {"A", "B"}
    assert tips["A"].shape == (50, len(root))
    ident_a = (tips["A"] == root).mean()
    ident_b = (tips["B"] == root).mean()
    assert ident_a > ident_b
