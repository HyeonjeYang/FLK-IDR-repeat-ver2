"""Curate the expanded FLK homolog/ortholog-candidate set and compare the
three LEPQQYEV repeats across included species.

Data flow (see docs/METHODS.md and data/provenance/SOURCES.md):
- `data/raw/homolog_candidates_raw.fasta` -- every candidate considered,
  independently retrieved from UniProtKB, immutable.
- `data/provenance/homolog_provenance.tsv` -- one row per candidate,
  recording accession/database/length/SHA256/orthology status/KH-domain
  status/N-terminal status/inclusion decision and reason, and each
  candidate's own KH1-domain start (the conserved structural anchor used
  to define its homologous N-terminal region).
- `data/curated/flk_homologs_curated.fasta` -- only rows with
  included == True, ungapped, generated from the two files above.

Repeat mapping does not depend on any pre-computed multiple-sequence
alignment. Each candidate's own KH1 start (a UniProt-annotated Domain
feature, or -- for the one candidate whose proteome carries no feature
annotation -- a fixed offset from an exact match to a conserved 31-aa
KH1-flanking sequence fingerprint, see homolog_provenance.tsv) anchors a
pairwise (Needleman-Wunsch, BLOSUM62) alignment of that candidate's own
pre-KH1 region against A. thaliana's, from which A. thaliana's three
repeat positions are mapped onto the candidate.
"""
from .blosum62 import BLOSUM62, alignment_score, self_score, GAP_SCORE
from .architecture import MOTIF, window_composition

CURATED_STATUSES = {"REFERENCE", "HIGH_CONFIDENCE_ORTHOLOG", "SUPPORTED_ORTHOLOG_CANDIDATE", "UNCERTAIN"}


def parse_fasta(path) -> dict:
    """Minimal FASTA parser. Keys are the full header line (text after
    '>'), preserving record order (dict, Python 3.7+)."""
    seqs, name, buf = {}, None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name = line[1:].strip()
            buf = []
        else:
            buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def parse_raw_candidates(path) -> dict:
    """{accession: ungapped sequence} from the raw candidate FASTA, whose
    headers are 'ACCESSION|species|length=N'."""
    raw = parse_fasta(path)
    return {header.split("|")[0]: seq for header, seq in raw.items()}


def needleman_wunsch(ref: str, query: str, gap_penalty: int = GAP_SCORE) -> tuple[str, str]:
    """Global pairwise alignment (BLOSUM62, linear gap penalty), returning
    two equal-length gapped strings. Pure-Python DP; fine for the short
    (<= ~200 aa) pre-KH1 regions this module aligns."""
    n, m = len(ref), len(query)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + BLOSUM62[(ri, query[j - 1])]
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(match, delete, insert)

    ref_aln, query_aln = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + BLOSUM62[(ref[i - 1], query[j - 1])]:
            ref_aln.append(ref[i - 1]); query_aln.append(query[j - 1]); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap_penalty:
            ref_aln.append(ref[i - 1]); query_aln.append("-"); i -= 1
        else:
            ref_aln.append("-"); query_aln.append(query[j - 1]); j -= 1
    return "".join(reversed(ref_aln)), "".join(reversed(query_aln))


def ungapped_position_to_column(gapped: str) -> dict:
    """{1-based ungapped position: 0-based alignment column} for one
    gapped row (e.g. the reference side of a pairwise alignment)."""
    col_of_pos = {}
    pos = 0
    for col, ch in enumerate(gapped):
        if ch != "-":
            pos += 1
            col_of_pos[pos] = col
    return col_of_pos


def build_curated_fasta(raw_seqs: dict, provenance_rows: list[dict], out_path) -> dict:
    """Write the curated (ungapped, included-only) FASTA and return
    {accession: sequence}. Sequences are copied verbatim from the raw
    candidate set -- never edited, trimmed, or reordered beyond filtering."""
    curated = {}
    lines = []
    for row in provenance_rows:
        if not row["included"]:
            continue
        acc = row["accession"]
        seq = raw_seqs[acc]
        curated[acc] = seq
        header = (f">{acc}|{row['species']}|status={row['orthology_status']}|length={len(seq)}")
        lines.append(header)
        for i in range(0, len(seq), 60):
            lines.append(seq[i:i + 60])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return curated


def repeat_comparison_table(raw_seqs: dict, provenance_rows: list[dict],
                             athaliana_repeat_starts: list[int], athaliana_kh1_start: int) -> list[dict]:
    """For every included candidate, pairwise-align its own pre-KH1 region
    against A. thaliana's, map the three repeat positions through that
    alignment, and report exact identity, BLOSUM62 similarity, indels, and
    local (pre-KH1) E/Q/Y composition."""
    athal_seq = raw_seqs["Q9SR13"]
    athal_region = athal_seq[:athaliana_kh1_start - 1]

    rows = []
    for row in provenance_rows:
        if not row["included"] or row["accession"] == "Q9SR13":
            continue
        acc = row["accession"]
        seq = raw_seqs[acc]
        kh1_start = row["kh1_start_1based"]
        query_region = seq[:int(kh1_start) - 1] if kh1_start not in ("", None) else seq

        ref_aln, query_aln = needleman_wunsch(athal_region, query_region)
        col_of_pos = ungapped_position_to_column(ref_aln)
        local_comp = window_composition(query_region, 1, len(query_region)) if query_region else None

        for repeat_idx, start in enumerate(athaliana_repeat_starts, start=1):
            cols = [col_of_pos[start + i] for i in range(len(MOTIF))]
            window = "".join(query_aln[c] for c in cols)
            has_gap = "-" in window
            n_sub = sum(1 for a, b in zip(window, MOTIF) if a != b and a != "-")
            exact_identity = (not has_gap) and n_sub == 0
            rows.append({
                "species": row["species"],
                "accession": acc,
                "orthology_status": row["orthology_status"],
                "repeat_index": repeat_idx,
                "reference_sequence": MOTIF,
                "aligned_sequence": window,
                "exact_identity": exact_identity,
                "n_substitutions_vs_reference": n_sub,
                "has_indel": has_gap,
                "confidently_alignable": not has_gap,
                "blosum62_alignment_score": alignment_score(window, MOTIF),
                "blosum62_self_score_reference": self_score(MOTIF),
                "blosum62_similarity": max(0.0, alignment_score(window, MOTIF) / self_score(MOTIF)),
                "local_region_upstream_of_KH1_length": local_comp["length"] if local_comp else 0,
                "local_E_fraction": local_comp["E_fraction"] if local_comp else float("nan"),
                "local_Q_fraction": local_comp["Q_fraction"] if local_comp else float("nan"),
                "local_Y_fraction": local_comp["Y_fraction"] if local_comp else float("nan"),
                "local_EQY_fraction": local_comp["EQY_fraction"] if local_comp else float("nan"),
            })
    return rows
