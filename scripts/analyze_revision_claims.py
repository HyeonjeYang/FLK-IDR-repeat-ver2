#!/usr/bin/env python3
"""Reproducible descriptive checks requested in the manuscript revision.

Outputs are deliberately descriptive. The close-relative position controls
share ancestry and the repeat coordinates were recognized before this test.
The proteome background is not matched for composition or ancestry.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from random import Random
import sys

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from flk_repeat.homologs import parse_raw_candidates, needleman_wunsch, ungapped_position_to_column  # noqa: E402
from flk_repeat.paths import RAW_HOMOLOG_CANDIDATES_FASTA  # noqa: E402
from flk_repeat.repeats import best_repeat_family, motif_family_copies  # noqa: E402
from scripts.build_idr_background import disordered_regions  # noqa: E402

OUT = ROOT / "results" / "expanded"
MOTIF = "LEPQQYEV"
REPEAT_STARTS = (51, 63, 75)
REPEAT_SITES = {i for start in REPEAT_STARTS for i in range(start, start + 8)}
MSA = ROOT / "data" / "validation" / "flk_full_length_clustalo.fasta"


def exact_kmer_counts(seq: str, k: int = 8) -> tuple[int, str, int]:
    positions = defaultdict(list)
    for i in range(len(seq) - k + 1):
        positions[seq[i:i+k]].append(i)
    best_exact = sorted(((len(v), key) for key, v in positions.items()),
                        key=lambda item: (-item[0], item[1]))[0]
    best_nonoverlap = 0
    for starts in positions.values():
        count, previous = 0, -k
        for start in starts:
            if start >= previous + k:
                count += 1
                previous = start
        best_nonoverlap = max(best_nonoverlap, count)
    return best_exact[0], best_exact[1], best_nonoverlap


def complex_kmer_counts(seq: str, min_distinct: int, k: int = 8) -> tuple[int, int]:
    """Maximum exact k-mer multiplicity after a descriptive complexity cutoff."""
    positions = defaultdict(list)
    for i in range(len(seq) - k + 1):
        word = seq[i:i+k]
        if len(set(word)) >= min_distinct:
            positions[word].append(i)
    best_overlap = max((len(v) for v in positions.values()), default=0)
    best_nonoverlap = 0
    for starts in positions.values():
        count, previous = 0, -k
        for start in starts:
            if start >= previous + k:
                count += 1
                previous = start
        best_nonoverlap = max(best_nonoverlap, count)
    return best_overlap, best_nonoverlap


def expanded_repeat_table() -> pd.DataFrame:
    audit = pd.read_csv(OUT / "reciprocal_blast_audit.tsv", sep="\t").set_index("query_accession")
    representatives = {r.id: str(r.seq) for r in SeqIO.parse(
        ROOT / "data" / "expanded" / "sequences" / "representative_40.fasta", "fasta")}
    rows = []
    for acc, rep in audit.iterrows():
        if not bool(rep["passes_repeat_region_qc"]):
            continue
        assert acc in representatives
        region = representatives[acc][:int(rep["pre_KH1_length"])]
        t8, seed, t8_no = exact_kmer_counts(region)
        near = motif_family_copies(region, MOTIF, 3)
        family = best_repeat_family(region, 8, 1)
        rows.append({"species": rep["species"], "accession": acc,
                     "pre_KH1_length": len(region),
                     "LEPQQYEV_exact_copies": region.count(MOTIF),
                     "LEPQQYEV_within_3_substitutions_nonoverlap": len(near),
                     "LEPQQYEV_near_windows": ";".join(f"{p}:{w}:{d}" for p, w, d in near),
                     "T8_exact_overlapping": t8,
                     "T8_exact_nonoverlapping": t8_no,
                     "best_exact_8mer": seed,
                     "best_1_mismatch_family_nonoverlapping": family["n_copies_nonoverlapping"]})
    df = pd.DataFrame(rows).sort_values("species")
    assert len(df) == 26
    df.to_csv(OUT / "qc_passed_repeat_descriptive.tsv", sep="\t", index=False)
    return df


def close_relative_retention() -> pd.DataFrame:
    seqs = parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    ref_seq = seqs["Q9SR13"]
    msa = {r.id: str(r.seq) for r in SeqIO.parse(MSA, "fasta")}
    msa_columns = ungapped_position_to_column(msa["Q9SR13"])
    candidates = [("Arabidopsis lyrata", "D7L277", 176),
                  ("Arabidopsis halleri", "A0ACE2D236", 183),
                  ("Arabidopsis suecica (divergent copy)", "A0A8T2B1P0", 185)]
    rows, vectors, msa_vectors = [], {}, {}
    for name, acc, kh1 in candidates:
        a, b = needleman_wunsch(ref_seq[:185], seqs[acc][:kh1 - 1])
        pair_cols = ungapped_position_to_column(a)
        pair = [int(ref_seq[i-1] != b[pair_cols[i]]) for i in range(1, 151)]
        multiple = [int(ref_seq[i-1] != msa[acc][msa_columns[i]]) for i in range(1, 151)]
        assert (sum(pair[i-1] for i in REPEAT_SITES), sum(pair)) == (sum(multiple[i-1] for i in REPEAT_SITES), sum(multiple)), name
        vectors[acc] = pair
        msa_vectors[acc] = multiple
        repeat_diff = sum(pair[i-1] for i in REPEAT_SITES)
        flank_diff = sum(pair) - repeat_diff
        rows.append({"species": name, "accession": acc,
                     "repeat_differences": repeat_diff, "repeat_sites": 24,
                     "flank_differences": flank_diff, "flank_sites": 126,
                     "alignment_methods_agree_counts": True,
                     "pairwise_MSA_site_call_differences": sum(x != y for x, y in zip(pair, multiple))})
    df = pd.DataFrame(rows)
    assert list(df.repeat_differences) == [1, 1, 0]
    assert list(df.flank_differences) == [26, 18, 16]
    df.to_csv(OUT / "close_relative_retention.tsv", sep="\t", index=False)

    # The same reference-coordinate triplet is evaluated across all three
    # candidates, retaining shared changes rather than treating species as
    # independent replicates. Control triplets cannot overlap the motif.
    control = []
    for first in range(1, 120):
        sites = {i for start in (first, first + 12, first + 24)
                 for i in range(start, start + 8)}
        if sites & REPEAT_SITES:
            continue
        control.append({"first_window_start": first,
                        "pairwise_total_differences_three_species": sum(v[i-1] for v in vectors.values() for i in sites),
                        "MSA_total_differences_three_species": sum(v[i-1] for v in msa_vectors.values() for i in sites)})
    cdf = pd.DataFrame(control)
    assert len(cdf) == 56
    cdf.to_csv(OUT / "close_relative_spacing_controls.tsv", sep="\t", index=False)
    return df


def outgroup_polarization_and_isoforms() -> None:
    """Audit direction of flank differences and alternative protein lengths."""
    seqs = parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    msa = {r.id: str(r.seq) for r in SeqIO.parse(MSA, "fasta")}
    ref = msa["Q9SR13"]
    columns = ungapped_position_to_column(ref)
    rows = []
    for species, acc in [("Arabidopsis lyrata", "D7L277"),
                         ("Arabidopsis halleri", "A0ACE2D236"),
                         ("Arabidopsis suecica", "A0A8T2B1P0")]:
        counts = Counter()
        for pos in range(1, 151):
            if pos in REPEAT_SITES:
                continue
            col = columns[pos]
            ath, query = ref[col], msa[acc][col]
            if ath == query:
                continue
            capsella, camelina = msa["R0G4A9"][col], msa["A0ABM0SVE8"][col]
            if capsella == camelina == query:
                counts["thaliana_branch"] += 1
            elif capsella == camelina == ath:
                counts["other_branch"] += 1
            else:
                counts["unresolved"] += 1
        rows.append({"species": species, "accession": acc,
                     "flank_thaliana_branch": counts["thaliana_branch"],
                     "flank_other_branch": counts["other_branch"],
                     "flank_unresolved": counts["unresolved"],
                     "repeat_expected_lower_24_sites": 24 * counts["other_branch"] / 126,
                     "repeat_expected_upper_24_sites": 24 * (counts["other_branch"] + counts["unresolved"]) / 126})
    out = pd.DataFrame(rows)
    assert list(zip(out.flank_thaliana_branch, out.flank_other_branch, out.flank_unresolved)) == [(9, 9, 8), (8, 3, 7), (8, 2, 6)]
    out.to_csv(OUT / "close_relative_flank_polarization.tsv", sep="\t", index=False)

    # Deleting positions 51--74 is one valid 24-aa alignment of the tandem
    # region. Equivalent alignments can assign the deletion to other copies.
    short_ath = seqs["A0A1I9LM90"]
    short_arenosa = seqs["A0A8S1ZSW1"]
    full_ath = seqs["Q9SR13"]
    full_suecica = seqs["A0A8T2B1P0"]
    assert full_ath[:50] + full_ath[74:] == short_ath
    deletion_starts = [i + 1 for i in range(len(full_ath) - 23)
                       if full_ath[:i] + full_ath[i + 24:] == short_ath]
    assert deletion_starts == list(range(51, 60))
    assert len(short_ath) == len(short_arenosa) == 553
    assert sum(a != b for a, b in zip(full_suecica[:50] + full_suecica[74:], short_arenosa)) == 3
    assert sum(a != b for a, b in zip(full_ath[:50] + full_ath[74:], short_arenosa)) == 25
    iso = pd.DataFrame([
        {"species": "Arabidopsis thaliana", "accession": "Q9SR13", "length": 577, "exact_LEPQQYEV_copies": full_ath.count(MOTIF), "comparison": "reference full isoform"},
        {"species": "Arabidopsis thaliana", "accession": "A0A1I9LM90", "length": 553, "exact_LEPQQYEV_copies": short_ath.count(MOTIF), "comparison": "Q9SR13 protein with 24-aa deletion starting at any residue 51-59; Col-0 annotated splice model audited separately"},
        {"species": "Arabidopsis suecica", "accession": "A0A8T2B1P0", "length": 577, "exact_LEPQQYEV_copies": full_suecica.count(MOTIF), "comparison": "full-length selected copy"},
        {"species": "Arabidopsis arenosa", "accession": "A0A8S1ZSW1", "length": 553, "exact_LEPQQYEV_copies": short_arenosa.count(MOTIF), "comparison": "3 differences from B1P0 after the same admissible 24-aa deletion; 25 from Q9SR13"},
    ])
    iso.to_csv(OUT / "short_isoform_array_audit.tsv", sep="\t", index=False)


def idr_window_background() -> pd.DataFrame:
    rng = Random(20261101)
    rows = []
    for region in disordered_regions():
        if len(region) < 150:
            continue
        start = rng.randrange(len(region) - 149)
        window = region[start:start + 150]
        t8, seed, t8_no = exact_kmer_counts(window)
        c5, c5_no = complex_kmer_counts(window, 5)
        c6, c6_no = complex_kmer_counts(window, 6)
        rows.append({"region_sha256": sha256(region.encode()).hexdigest(),
                     "region_length": len(region), "window_start_1based": start + 1,
                     "T8_exact_overlapping": t8,
                     "T8_exact_nonoverlapping": t8_no,
                     "best_exact_8mer": seed,
                     "T8_min5_distinct_overlapping": c5,
                     "T8_min5_distinct_nonoverlapping": c5_no,
                     "T8_min6_distinct_overlapping": c6,
                     "T8_min6_distinct_nonoverlapping": c6_no,
                     "LEPQQYEV_exact_copies": window.count(MOTIF)})
    df = pd.DataFrame(rows)
    assert len(df) == 883
    df.to_csv(OUT / "athaliana_idr_150aa_window_repeats.tsv", sep="\t", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("close", "expanded", "background", "all"),
                        default="all", help="close runs from compact public inputs; all also needs the large source downloads")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.section in ("expanded", "all"):
        expanded = expanded_repeat_table()
        print(f"26 QC-passing species: exact motif in {(expanded.LEPQQYEV_exact_copies > 0).sum()}; "
              f"T8>=3 in {(expanded.T8_exact_overlapping >= 3).sum()}")
    if args.section in ("close", "all"):
        close = close_relative_retention()
        outgroup_polarization_and_isoforms()
        print("close-relative repeat differences:", list(close.repeat_differences),
              "flank differences:", list(close.flank_differences))
    if args.section in ("background", "all"):
        bg = idr_window_background()
        print(f"150-aa IDR windows: {len(bg)}; overlapping T8>=3 in "
              f"{(bg.T8_exact_overlapping >= 3).sum()}; non-overlapping T8>=3 in "
              f"{(bg.T8_exact_nonoverlapping >= 3).sum()}")
        print(f"complexity >=5 distinct residues: overlapping >=3 in "
              f"{(bg.T8_min5_distinct_overlapping >= 3).sum()}, non-overlapping >=3 in "
              f"{(bg.T8_min5_distinct_nonoverlapping >= 3).sum()}")


if __name__ == "__main__":
    main()
