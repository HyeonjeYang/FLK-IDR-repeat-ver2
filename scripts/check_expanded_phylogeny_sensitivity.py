#!/usr/bin/env python3
"""Sensitivity tree using all well-occupied FLK-aligned protein columns."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flk_repeat import phylo  # noqa: E402
from flk_repeat.paths import RESULTS_DIR  # noqa: E402
from analyze_expanded_phylogeny import _source_records, FLK, PARALOGS  # noqa: E402

OUT = RESULTS_DIR / "expanded"


def main(bootstraps: int = 400) -> pd.DataFrame:
    tips = [r.id for r in SeqIO.parse(OUT / "expanded_focus_kh_core_alignment.fasta", "fasta")]
    qc, records, _ = _source_records()
    msa = phylo.align({a: records[a] for a in tips}, threads=0)
    arr = np.array([list(s) for s in msa.values()])
    present = np.isin(arr, list(phylo.AA))
    ref_index = list(msa).index(FLK)
    selected = (present.mean(axis=0) >= 0.75) & present[ref_index]
    if selected.sum() < 150:
        raise AssertionError(f"Only {selected.sum()} high-occupancy columns")
    aln = {a: "".join(arr[i, selected]) for i, a in enumerate(msa)}
    (OUT / "expanded_focus_occupancy_alignment.fasta").write_text(
        "".join(f">{a}\n{s}\n" for a, s in aln.items()), encoding="utf-8")
    tree = phylo.bootstrap_nj(aln, n_bootstrap=bootstraps, seed=20261102)
    members, support = phylo.flk_clade(tree, FLK, PARALOGS)
    rooted = phylo.rooted_tree(tree["edges"], tree["names"], "F4KDN0")
    (OUT / "expanded_focus_occupancy_tree.nwk").write_text(
        phylo.to_newick(rooted) + "\n", encoding="utf-8")
    selected_genes = qc[qc.is_initial_selected_gene].copy()
    selected_genes["in_occupancy_FLK_side"] = selected_genes.tree_accession.isin(members)
    selected_genes["occupancy_FLK_side_bootstrap_pct"] = support
    selected_genes[["species", "tree_accession", "in_occupancy_FLK_side",
                    "occupancy_FLK_side_bootstrap_pct"]].to_csv(
        OUT / "expanded_phylogeny_sensitivity.tsv", sep="\t", index=False)
    print(f"{selected.sum()} high-occupancy columns; {len(tree['names'])} tips; "
          f"FLK-side split support {support:.1f}%; "
          f"{selected_genes.in_occupancy_FLK_side.sum()}/40 initial candidates on FLK side")
    return selected_genes


if __name__ == "__main__":
    main()
