#!/usr/bin/env python3
"""Exploratory angiosperm repeat survey from the cached Ensembl Compara JSON.

This is an independent, resource-limited pilot while NCBI BLAST source
records are pending. The fixed TARGETS list was written for the revision's
BLAST survey; here its species names are matched to Compara species IDs.
Candidates are chosen by sequence QC and highest full-protein identity,
without looking at N-terminal repeats. Compara homology is evidence of a
candidate, not sufficient by itself for the planned orthology call.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flk_repeat.expanded import locate_kh1, athaliana_reference  # noqa: E402
from flk_repeat.features import STANDARD_AA  # noqa: E402
from flk_repeat.repeats import best_repeat_family, motif_family_copies  # noqa: E402
from flk_repeat.paths import DATA_DIR, RESULTS_DIR, FIGURES_DIR  # noqa: E402
from flk_repeat.figure_labels import scientific_name  # noqa: E402
from select_expanded_species import TARGETS  # noqa: E402

SOURCE = DATA_DIR / "expanded" / "ensembl" / "compara_plants_AT3G04610_orthologues.json"
OUT = RESULTS_DIR / "expanded" / "compara_pilot_repeats.tsv"
AUDIT = RESULTS_DIR / "expanded" / "compara_pilot_selection.tsv"
FIG = FIGURES_DIR / "figS4_compara_pilot.png"


def _species_id(name: str, available: set[str]) -> str | None:
    key = "_".join(name.lower().split()[:2])
    matches = sorted(x for x in available if x == key or x.startswith(key + "_"))
    return matches[0] if matches else None


def analyze() -> tuple[pd.DataFrame, pd.DataFrame]:
    doc = json.loads(SOURCE.read_text(encoding="utf-8"))
    homologies = doc["data"][0]["homologies"]
    available = {x["target"]["species"] for x in homologies}
    ath_seq, ath_kh1 = athaliana_reference()
    rows, audit = [], []
    used = set()
    for lineage, order, names, reason in TARGETS:
        choice = next(((n, s) for n in names if (s := _species_id(n, available)) and s not in used), None)
        if not choice:
            audit.append({"target": names[0], "lineage": lineage, "compara_species": "",
                          "candidate_protein": "", "passed_qc": False, "decision": "species absent"})
            continue
        display_species, sid = choice
        used.add(sid)
        candidates = [x for x in homologies if x["target"]["species"] == sid]
        eligible = []
        for h in candidates:
            t = h["target"]
            seq = t.get("seq") or ""
            issues = []
            if not seq.startswith("M"):
                issues.append("missing start M")
            if not set(seq) <= set(STANDARD_AA):
                issues.append("nonstandard residue")
            if not 350 <= len(seq) <= 750:
                issues.append("length outside 350-750")
            kh = locate_kh1(seq, ath_seq, ath_kh1) if len(seq) >= 80 and set(seq) <= set(STANDARD_AA) else None
            if kh and kh["kh1_fraction_aligned"] < 0.8:
                issues.append("KH1 coverage below 0.8")
            if kh and kh["kh1_start_1based"] < 76:
                issues.append("pre-KH1 region below 75 aa")
            rec = {"target": names[0], "lineage": lineage, "compara_species": sid,
                   "candidate_protein": t.get("protein_id", ""), "homology_type": h["type"],
                   "target_identity_percent": t.get("perc_id"), "length": len(seq),
                   "kh1_start_1based": kh["kh1_start_1based"] if kh else None,
                   "kh1_fraction_aligned": kh["kh1_fraction_aligned"] if kh else None,
                   "passed_qc": not issues, "decision": "; ".join(issues) if issues else "eligible"}
            audit.append(rec)
            if not issues:
                eligible.append((float(t.get("perc_id") or 0), h, kh))
        if not eligible:
            continue
        eligible.sort(key=lambda x: (-x[0], x[1]["target"]["protein_id"]))
        ident, chosen, kh = eligible[0]
        t = chosen["target"]
        for a in audit:
            if a.get("candidate_protein") == t["protein_id"] and a.get("compara_species") == sid:
                a["decision"] = "selected: highest identity among QC-passing candidates"
        region = t["seq"][:kh["kh1_start_1based"] - 1]
        one = best_repeat_family(region, 8, 1)
        two = best_repeat_family(region, 8, 2)
        exact = best_repeat_family(region, 8, 0)
        motif = motif_family_copies(region, "LEPQQYEV", 3)
        rows.append({"lineage": lineage, "order": order,
                     "species": display_species,
                     "protein_id": t["protein_id"], "homology_type": chosen["type"],
                     "target_identity_percent": ident, "n_candidates": len(candidates),
                     "identity_margin_to_next_qc_candidate": ident - eligible[1][0] if len(eligible) > 1 else None,
                     "full_length": len(t["seq"]), "pre_KH1_length": len(region),
                     "T8_exact_overlapping": exact["n_overlapping_matches"],
                     "best_exact_8mer": exact["seed"],
                     "best_8mer_1_mismatch_seed": one["seed"],
                     "A8_1_overlapping": one["n_overlapping_matches"],
                     "best_8mer_1_mismatch_nonoverlapping_copies": one["n_copies_nonoverlapping"],
                     "best_8mer_2_mismatch_nonoverlapping_copies": two["n_copies_nonoverlapping"],
                     "LEPQQYEV_family_3_mismatch_nonoverlapping_copies": len(motif),
                     "verification_status": "COMPARA_ONLY_UNVERIFIED"})
    return pd.DataFrame(rows), pd.DataFrame(audit)


def make_figure(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, max(7, 0.29 * len(df) + 2.3)), constrained_layout=True)
    y = range(len(df))
    seed_x = max(4.7, df[["T8_exact_overlapping",
                         "best_8mer_1_mismatch_nonoverlapping_copies"]].to_numpy().max() + 0.7)
    for i, r in df.reset_index(drop=True).iterrows():
        color = "#2166ac" if r.lineage == "Monocots" else "#d6604d"
        ax.plot([r.T8_exact_overlapping, r.best_8mer_1_mismatch_nonoverlapping_copies], [i, i],
                color="#bbbbbb", lw=1)
        ax.scatter(r.T8_exact_overlapping, i, marker="s", s=36, color=color, zorder=3)
        ax.scatter(r.best_8mer_1_mismatch_nonoverlapping_copies, i, marker="o", s=36,
                   facecolors="white", edgecolors=color, linewidths=1.5, zorder=3)
        ax.text(seed_x, i, r.best_8mer_1_mismatch_seed, fontsize=7.5,
                family="monospace", va="center")
    ax.set_yticks(list(y), [scientific_name(s) for s in df.species], fontsize=8)
    ax.set_xlim(0.5, seed_x + 1.6)
    ax.set_xticks(range(1, int(seed_x)))
    ax.set_ylim(len(df) - 0.5, -1.1)
    ax.text(seed_x, -0.7, "Best ≤1-mismatch\nfamily seed", fontsize=8,
            ha="left", va="center")
    ax.set_xlabel("8-mer count (filled square: exact overlapping T8; open circle: ≤1-mismatch nonoverlapping family)")
    ax.set_title("Ensembl Compara pilot: pre-KH1 8-mer counts",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(FIG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> pd.DataFrame:
    df, audit = analyze()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    audit.to_csv(AUDIT, sep="\t", index=False)
    make_figure(df)
    print(f"Compara pilot: {len(df)} selected of {len(TARGETS)} targets; wrote {OUT}, {AUDIT}, {FIG}")
    return df


if __name__ == "__main__":
    main()
