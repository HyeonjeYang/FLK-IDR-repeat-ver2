"""Shared helpers for the expanded angiosperm FLK candidate set (revision,
Reviewer 1 comment 1): parsing cached NCBI BLAST JSON output and the
provenance/evidence table that records every orthology and
sequence-verification check per candidate.
"""
import json
import re
from pathlib import Path

from Bio import Align
from Bio.Align import substitution_matrices

from .paths import DATA_DIR, RESULTS_DIR

EXPANDED_DATA = DATA_DIR / "expanded"
EXPANDED_RESULTS = RESULTS_DIR / "expanded"
BLAST_DIR = EXPANDED_DATA / "blast"
BLAST_PARTITIONS = ["monocots", "eudicots_non_brassicaceae", "other_angiosperms", "brassicaceae"]

_ORGANISM_IN_TITLE = re.compile(r"\[([^\[\]]+)\]\s*$")


def parse_blast_json(path: Path) -> list[dict]:
    """One dict per subject sequence (best HSP), from NCBI JSON2_S output.
    Query coverage is the union of the query spans of all HSPs."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    search = doc["BlastOutput2"][0]["report"]["results"]["search"]
    qlen = search["query_len"]
    rows = []
    for hit in search.get("hits", []):
        desc = hit["description"][0]
        # NCBI's JSON `accession` omits the version (XP_....), whereas the
        # `id` and GenPept VERSION fields retain it (XP_.....1). Preserve
        # the exact subject version so downstream records join reliably.
        accession = desc["accession"]
        subject_id = desc.get("id", "")
        id_fields = subject_id.split("|")
        if len(id_fields) >= 3 and id_fields[1].startswith(accession + "."):
            accession = id_fields[1]
        hsps = hit["hsps"]
        best = max(hsps, key=lambda h: h["bit_score"])
        covered = set()
        for h in hsps:
            covered.update(range(min(h["query_from"], h["query_to"]), max(h["query_from"], h["query_to"]) + 1))
        m = _ORGANISM_IN_TITLE.search(desc.get("title", ""))
        rows.append({
            "accession": accession,
            "title": desc.get("title", ""),
            "taxid": desc.get("taxid"),
            "sciname": desc.get("sciname") or (m.group(1) if m else ""),
            "subject_length": hit["len"],
            "bit_score": best["bit_score"],
            "evalue": best["evalue"],
            "identity_fraction": best["identity"] / best["align_len"],
            "query_coverage": len(covered) / qlen,
            "best_hsp_query_from": best["query_from"],
            "best_hsp_query_to": best["query_to"],
            "best_hsp_hit_from": best["hit_from"],
            "best_hsp_hit_to": best["hit_to"],
        })
    return rows


def _local_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


def locate_kh1(seq: str, athaliana_seq: str, athaliana_kh1: tuple[int, int]) -> dict:
    """Project the A. thaliana KH1 domain (1-based inclusive coordinates)
    onto `seq` by local alignment (BLOSUM62, gap open 11 / extend 1, as in
    BLASTP). Returns the 1-based KH1 start in `seq`, the alignment score and
    the fraction of KH1 columns aligned."""
    start, end = athaliana_kh1
    domain = athaliana_seq[start - 1:end]
    aln = _local_aligner().align(domain, seq)[0]
    (d_blocks, s_blocks) = aln.aligned
    first_domain_pos, first_seq_pos = d_blocks[0][0], s_blocks[0][0]
    covered = sum(b - a for a, b in d_blocks)
    return {
        "kh1_start_1based": int(first_seq_pos - first_domain_pos + 1),
        "kh1_alignment_score": float(aln.score),
        "kh1_fraction_aligned": covered / len(domain),
    }


def load_all_blast_hits() -> list[dict]:
    rows = []
    for part in BLAST_PARTITIONS:
        path = BLAST_DIR / f"{part}.json"
        for r in parse_blast_json(path):
            r["partition"] = part
            rows.append(r)
    return rows


def parse_genpept(text: str) -> dict:
    """{accession.version: {gene_id, gene, organism, definition, length,
    sequence}} from concatenated GenPept flat-file records."""
    out = {}
    for rec in text.split("\n//"):
        version = re.search(r"^VERSION\s+(\S+)", rec, re.M)
        if not version:
            continue
        gene_id = re.search(r'/db_xref="GeneID:(\d+)"', rec)
        gene = re.search(r'/gene="([^"]+)"', rec)
        organism = re.search(r"^\s+ORGANISM\s+(.+)$", rec, re.M)
        definition = re.search(r"^DEFINITION\s+(.+?)(?=^\S)", rec, re.M | re.S)
        origin = rec.split("ORIGIN", 1)[1] if "ORIGIN" in rec else ""
        seq = re.sub(r"[^A-Za-z]", "", origin).upper()
        out[version.group(1)] = {
            "gene_id": gene_id.group(1) if gene_id else "",
            "gene": gene.group(1) if gene else "",
            "organism": organism.group(1).strip() if organism else "",
            "definition": re.sub(r"\s+", " ", definition.group(1)).strip() if definition else "",
            "length": len(seq),
            "sequence": seq,
        }
    return out


def athaliana_reference() -> tuple[str, tuple[int, int]]:
    from .sequence import load_flk_record
    rec = load_flk_record()
    kh1 = rec["provenance"]["kh_domains_1based_inclusive"][0]
    return rec["sequence"], (kh1["start"], kh1["end"])


def choose_representatives(isoform_records: dict | None = ...) -> list[dict]:
    """One FLK candidate per selected species: the gene of the best-scoring
    BLAST hit, and among that gene's RefSeq isoforms the one with the
    longest region upstream of KH1 (ties: longer protein, then accession).
    `isoform_records=...` (default) reads the cached isoform GenPept file
    if present; pass None to use only the BLAST hit records."""
    import pandas as pd

    hits = pd.read_csv(EXPANDED_DATA / "selection" / "candidate_hits.tsv", sep="\t")
    gp = parse_genpept((EXPANDED_DATA / "sequences" / "candidate_hits.gp").read_text(encoding="utf-8"))
    if isoform_records is ...:
        iso_path = EXPANDED_DATA / "sequences" / "representative_gene_isoforms.gp"
        isoform_records = parse_genpept(iso_path.read_text(encoding="utf-8")) if iso_path.exists() else {}
    records = {**gp, **(isoform_records or {})}
    ath_seq, ath_kh1 = athaliana_reference()

    reps = []
    for species, grp in hits.sort_values("bit_score", ascending=False).groupby("sciname", sort=True):
        top = grp.iloc[0]
        gene_id = records.get(top.accession, {}).get("gene_id", "")
        isoforms = [a for a, r in records.items() if gene_id and r["gene_id"] == gene_id] or [top.accession]
        scored = []
        for acc in isoforms:
            seq = records[acc]["sequence"]
            kh1 = locate_kh1(seq, ath_seq, ath_kh1)
            scored.append((-(kh1["kh1_start_1based"] - 1), -len(seq), acc, kh1))
        scored.sort()
        _, _, acc, kh1 = scored[0]
        reps.append({
            "species": species,
            "accession": acc,
            "gene_id": gene_id,
            "gene": records[acc]["gene"],
            "definition": records[acc]["definition"],
            "sequence": records[acc]["sequence"],
            "length": len(records[acc]["sequence"]),
            "top_blast_hit": top.accession,
            "top_blast_bit_score": float(top.bit_score),
            "n_isoforms_considered": len(isoforms),
            "isoform_accessions": ";".join(sorted(isoforms)),
            **kh1,
        })
    return reps
