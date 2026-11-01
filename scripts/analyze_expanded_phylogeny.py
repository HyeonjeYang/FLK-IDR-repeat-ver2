#!/usr/bin/env python3
"""Exploratory KH-domain gene-family trees for expanded RefSeq candidates.

The full tree includes one cached isoform per BLAST-supported gene. A
focused bootstrap tree includes the initially selected proteins, the
highest-scoring alternative gene per species, and up to two QC-passing
alternatives in each of the 14 species whose initial candidate fails
repeat-region QC. Only columns corresponding to the three annotated
A. thaliana FLK KH domains enter distance calculations, so low-complexity
N-terminal repeats cannot determine placement.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat import phylo  # noqa: E402
from flk_repeat.expanded import parse_genpept  # noqa: E402
from flk_repeat.paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from flk_repeat.sequence import load_flk_record  # noqa: E402

OUT = RESULTS_DIR / "expanded"
SEQ = DATA_DIR / "expanded" / "sequences"
KH_PARALOGS = {"Q9SR13": "FLK", "Q9SZH4": "PEP", "Q9LZ82": "BTR1",
               "F4KDN0": "HEN4", "Q8W4B1": "RCF3"}
FLK = "Q9SR13"
PARALOGS = [a for a in KH_PARALOGS if a != FLK]


def _source_records() -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    qc = pd.read_csv(OUT / "gene_candidate_qc.tsv", sep="\t")
    seqs = parse_genpept((SEQ / "candidate_hits.gp").read_text(encoding="utf-8"))
    seqs.update(parse_genpept((SEQ / "representative_gene_isoforms.gp").read_text(encoding="utf-8")))
    ath = pd.read_csv(DATA_DIR / "expanded" / "paralogs" / "athaliana_KH_reviewed.tsv", sep="\t")
    ath = ath[ath.Entry.isin(KH_PARALOGS)]
    if set(ath.Entry) != set(KH_PARALOGS):
        raise AssertionError("Missing A. thaliana FLK/paralog reference")
    records = {a: r["sequence"] for a, r in seqs.items()}
    records.update(dict(zip(ath.Entry, ath.Sequence)))
    if len(qc) != qc.tree_accession.nunique():
        raise AssertionError("Duplicate gene representatives in QC table")
    return qc, records, ath


def _core_alignment(records: dict[str, str]) -> dict[str, str]:
    msa = phylo.align(records, threads=0)
    ref = load_flk_record()
    domains = ref["provenance"]["kh_domains_1based_inclusive"]
    reference_positions = set()
    for domain in domains:
        reference_positions.update(range(domain["start"], domain["end"] + 1))
    idx = []
    residue = 0
    for col, aa in enumerate(msa[FLK]):
        if aa != "-":
            residue += 1
            if residue in reference_positions:
                idx.append(col)
    if len(idx) != len(reference_positions):
        raise AssertionError(f"KH core coverage {len(idx)} != {len(reference_positions)}")
    return {a: "".join(seq[i] for i in idx) for a, seq in msa.items()}


def _write_fasta(path: Path, seqs: dict[str, str]) -> None:
    path.write_text("".join(f">{a}\n{s}\n" for a, s in seqs.items()), encoding="utf-8")


def _tree_for(records: dict[str, str], name: str, bootstraps: int) -> tuple[dict, dict[str, float]]:
    print(f"{name}: aligning {len(records)} proteins", flush=True)
    core = _core_alignment(records)
    coverage = {a: sum(x != "-" for x in s) / len(s) for a, s in core.items()}
    low = [a for a, c in coverage.items() if c < 0.55 and a not in KH_PARALOGS]
    if low:
        print(f"{name}: excluding {len(low)} proteins with <55% FLK-KH core coverage", flush=True)
        core = {a: s for a, s in core.items() if a not in low}
    _write_fasta(OUT / f"{name}_kh_core_alignment.fasta", core)
    if bootstraps:
        print(f"{name}: {bootstraps} bootstrap NJ replicates", flush=True)
        tree = phylo.bootstrap_nj(core, n_bootstrap=bootstraps, seed=20261101)
    else:
        names, d = phylo.distance_matrix(core)
        edges = phylo.neighbor_joining(d)
        tree = {"names": names, "distances": d, "edges": edges,
                "support": {s: float("nan") for s in phylo.splits(edges, len(names))},
                "n_bootstrap": 0}
    rooted = phylo.rooted_tree(tree["edges"], tree["names"], "F4KDN0")
    (OUT / f"{name}_kh_tree.nwk").write_text(phylo.to_newick(rooted) + "\n", encoding="utf-8")
    print(f"{name}: wrote KH-core alignment and Newick tree", flush=True)
    return tree, coverage


def main(bootstraps: int = 200) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    qc, records, ath = _source_records()
    failed_species = set(qc.loc[qc.is_initial_selected_gene &
                                ~qc.passes_repeat_region_qc, "species"])
    full_accessions = list(qc.tree_accession) + list(KH_PARALOGS)
    full_records = {a: records[a] for a in full_accessions}
    full_tree, full_cov = _tree_for(full_records, "expanded_all_genes", 0)
    full_members, _ = phylo.flk_clade(full_tree, FLK, PARALOGS)

    focus = qc[qc.is_initial_selected_gene].copy()
    extra = qc[~qc.is_initial_selected_gene].sort_values(
        ["species", "top_blast_bit_score"], ascending=[True, False])
    focus = pd.concat([focus, extra.groupby("species").head(1)])
    rescue = extra[extra.species.isin(failed_species) & extra.passes_repeat_region_qc]
    focus = pd.concat([focus, rescue.groupby("species").head(2)])
    focus = focus.drop_duplicates("tree_accession")
    focus_accessions = list(focus.tree_accession) + list(KH_PARALOGS)
    focus_records = {a: records[a] for a in focus_accessions}
    focus_tree, focus_cov = _tree_for(focus_records, "expanded_focus", bootstraps)
    focus_members, focus_support = phylo.flk_clade(focus_tree, FLK, PARALOGS)
    full_idx = {name: i for i, name in enumerate(full_tree["names"])}
    flk_idx = full_idx[FLK]
    paralog_idx = [full_idx[p] for p in PARALOGS]

    rows = []
    for r in qc.itertuples(index=False):
        a = r.tree_accession
        in_focus = a in focus_tree["names"]
        distance_to_flk = float(full_tree["distances"][full_idx[a], flk_idx])
        distance_to_nearest_paralog = min(
            float(full_tree["distances"][full_idx[a], i]) for i in paralog_idx)
        rows.append({
            "species": r.species, "gene_id": r.gene_id,
            "repeat_representative_accession": r.representative_accession,
            "tree_accession": a,
            "is_initial_selected_gene": r.is_initial_selected_gene,
            "passes_repeat_region_qc": r.passes_repeat_region_qc,
            "top_blast_bit_score": r.top_blast_bit_score,
            "all_gene_tree_KH_coverage": full_cov[a],
            "KH_distance_to_athaliana_FLK": distance_to_flk,
            "KH_distance_to_nearest_athaliana_paralog": distance_to_nearest_paralog,
            "FLK_distance_advantage": distance_to_nearest_paralog - distance_to_flk,
            "in_all_gene_FLK_side": a in full_members,
            "included_in_focus_tree": in_focus,
            "in_focus_FLK_side": a in focus_members if in_focus else pd.NA,
            "focus_FLK_side_bootstrap_pct": focus_support if in_focus else pd.NA,
        })
    out = pd.DataFrame(rows).sort_values(["species", "top_blast_bit_score"],
                                         ascending=[True, False])
    out.to_csv(OUT / "expanded_phylogeny_audit.tsv", sep="\t", index=False)
    metadata = pd.concat([qc[["tree_accession", "species", "gene_id",
                              "is_initial_selected_gene", "passes_repeat_region_qc"]]
                          .rename(columns={"tree_accession": "tip"}),
                          pd.DataFrame([{"tip": a, "species": "Arabidopsis thaliana",
                                         "gene_id": KH_PARALOGS[a],
                                         "is_initial_selected_gene": False,
                                         "passes_repeat_region_qc": False}
                                        for a in KH_PARALOGS])], ignore_index=True)
    metadata.to_csv(OUT / "expanded_tree_tip_metadata.tsv", sep="\t", index=False)
    primary = out[out.is_initial_selected_gene]
    print(f"full tree: {len(full_tree['names'])} tips, "
          f"{sum(primary.in_all_gene_FLK_side)}/40 initial candidates on FLK side", flush=True)
    print(f"focus tree: {len(focus_tree['names'])} tips, FLK-side split support "
          f"{focus_support:.1f}%, {sum(primary.in_focus_FLK_side)}/40 initial candidates on FLK side",
          flush=True)
    print(f"wrote {OUT / 'expanded_phylogeny_audit.tsv'}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=200)
    args = parser.parse_args()
    main(args.bootstraps)
