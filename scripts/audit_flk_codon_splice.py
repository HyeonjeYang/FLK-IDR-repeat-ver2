#!/usr/bin/env python3
"""Reproduce FLK transcript, codon, and comparison-array audits offline.

Source GenBank records and retrieval details are in data/expanded/nucleotide/.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "expanded" / "nucleotide"
RESULTS = ROOT / "results" / "expanded"


def record(accession: str):
    return SeqIO.read(SOURCE / f"{accession}.gb", "genbank")


def cds_and_protein(rec) -> tuple[str, str]:
    feature = next(f for f in rec.features if f.type == "CDS")
    cds = str(feature.extract(rec.seq)).upper()
    protein = feature.qualifiers["translation"][0]
    assert str(Seq(cds).translate(to_stop=True)) == protein
    return cds, protein


def write_rows(name: str, rows: list[dict]) -> None:
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ay_cds, ay_aa = cds_and_protein(record("AY070475.1"))
    assert (len(ay_cds), len(ay_aa)) == (1734, 577)
    raw = list(SeqIO.parse(ROOT / "data" / "raw" / "homolog_candidates_raw.fasta", "fasta"))
    q9 = next(str(r.seq) for r in raw if r.id.startswith("Q9SR13|"))
    d7 = next(str(r.seq) for r in raw if r.id.startswith("D7L277|"))
    short_reference = next(str(r.seq) for r in raw if r.id.startswith("A0A1I9LM90|"))
    assert ay_aa == q9

    transcript_rows = []
    long_cdss, short_cdss = set(), set()
    for acc in ("NM_111333.5", "NM_001337535.1", "NM_001337536.1", "NM_001337537.1"):
        rec = record(acc)
        cds, aa = cds_and_protein(rec)
        feature = next(f for f in rec.features if f.type == "CDS")
        transcript_rows.append({"transcript": acc, "protein": feature.qualifiers["protein_id"][0],
                                "cds_nt": len(cds), "protein_aa": len(aa),
                                "same_as_AY070475_CDS": cds == ay_cds})
        (long_cdss if len(aa) == 577 else short_cdss).add(cds)
        if len(aa) == 553:
            assert aa == short_reference
    assert len(long_cdss) == len(short_cdss) == 1
    assert next(iter(long_cdss)) == ay_cds
    short_cds = next(iter(short_cdss))
    assert len(short_cds) == len(ay_cds) - 72
    protein_deletions = [i + 1 for i in range(len(ay_aa) - 23)
                         if ay_aa[:i] + ay_aa[i + 24:] == short_reference]
    assert protein_deletions == list(range(51, 60))
    write_rows("flk_transcript_model_audit.tsv", transcript_rows)

    genome = record("NC_003074.8_1250553_1254984")
    gene_cdss = {f.qualifiers["protein_id"][0]: f for f in genome.features if f.type == "CDS"}
    long_first = max(gene_cdss["NP_187112.1"].location.parts, key=len)
    short_parts = [p for p in gene_cdss["NP_001325771.1"].location.parts
                   if p.start >= long_first.start and p.end <= long_first.end]
    assert (int(long_first.start), int(long_first.end)) == (3021, 3720)
    assert sorted((int(p.start), int(p.end)) for p in short_parts) == [(3021, 3477), (3549, 3720)]
    intron = genome.seq[3477:3549].reverse_complement()
    assert len(intron) == 72 and str(intron[:2]) == "GT" and str(intron[-2:]) == "AG"
    write_rows("flk_splice_model_audit.tsv", [{"gene": "AT3G04610", "reference": "NC_003074.8",
        "genomic_start_1based": 1254030, "genomic_end_1based": 1254101,
        "intron_nt": len(intron), "transcript_strand_donor": str(intron[:2]),
        "transcript_strand_acceptor": str(intron[-2:]),
        "long_protein_aa": 577, "short_protein_aa": 553,
        "protein_alignment_deletion_starts_1based": ";".join(map(str, protein_deletions)),
        "junction_read_support": "not_assessed"}])

    lyr_protein = record("EFH60713.1")
    assert str(lyr_protein.seq) == d7
    coded_by = next(f for f in lyr_protein.features if f.type == "CDS").qualifiers["coded_by"][0]
    exons = [tuple(map(int, x)) for x in re.findall(r"GL348715\.1:(\d+)\.\.(\d+)", coded_by)]
    assert len(exons) == 6
    lyr_genome = record("GL348715.1_1584774_1588296")
    joined = "".join(str(lyr_genome.seq[a - 1584774:b - 1584774 + 1]) for a, b in exons)
    lyr_cds = str(Seq(joined).reverse_complement())
    assert str(Seq(lyr_cds).translate(to_stop=True)) == d7

    codon_rows = []
    for species, accession, cds, aa, starts in (
        ("Arabidopsis thaliana", "AY070475.1", ay_cds, ay_aa, (51, 63, 75)),
        ("Arabidopsis lyrata", "EFH60713.1", lyr_cds, d7, (45, 57, 69)),
    ):
        for index, start in enumerate(starts, 1):
            codons = [cds[3 * (pos - 1):3 * pos] for pos in range(start, start + 8)]
            motif = aa[start - 1:start + 7]
            assert "".join(str(Seq(c).translate()) for c in codons) == motif
            codon_rows.append({"species": species, "source_accession": accession,
                "copy": index, "aa_start_1based": start, "aa_end_1based": start + 7,
                "peptide": motif, "codons": " ".join(codons), "pro_codons": codons[2],
                "terminal_residue_codon": codons[7]})
    assert [r["pro_codons"] for r in codon_rows[:3]] == ["CCT", "CCA", "CCT"]
    assert [r["terminal_residue_codon"] for r in codon_rows[:3]] == ["GTA", "GTA", "GTT"]
    assert [r["terminal_residue_codon"] for r in codon_rows[3:]] == ["GTA", "CTA", "GTT"]
    write_rows("flk_repeat_codon_audit.tsv", codon_rows)

    selected = {r.id: str(r.seq) for r in SeqIO.parse(
        ROOT / "data" / "expanded" / "sequences" / "candidate_hits.fasta", "fasta")}
    motif_rows = []
    for species, accession, length, motif in (
        ("Gossypium raimondii", "XP_012492805.1", 149, "ANLEHIAN"),
        ("Malus domestica", "XP_008350637.2", 155, "ALAQFQAQ"),
    ):
        region = selected[accession][:length]
        starts = [i + 1 for i in range(len(region) - 7) if region[i:i + 8] == motif]
        assert len(starts) == 3 and all(b - a >= 8 for a, b in zip(starts, starts[1:]))
        motif_rows.append({"species": species, "accession": accession,
            "repeated_8mer": motif, "distinct_residue_types": len(set(motif)),
            "starts_1based": ";".join(map(str, starts)),
            "start_spacing_aa": ";".join(str(b - a) for a, b in zip(starts, starts[1:]))})
    write_rows("nonoverlap_candidate_motifs.tsv", motif_rows)
    print("Validated four Col-0 transcripts, canonical 72-nt splice model, 9 protein deletion placements,")
    print("six Arabidopsis repeat codon sequences, and Gossypium/Malus array spacing.")


if __name__ == "__main__":
    main()
