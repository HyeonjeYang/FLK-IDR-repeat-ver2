"""Repository-relative path constants. No absolute or user-specific paths."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
SEQUENCE_DIR = DATA_DIR / "sequence"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
PROVENANCE_DIR = DATA_DIR / "provenance"

# Legacy (superseded) homolog material from the prior repository's 8-record
# Clustal-Omega alignment. Retained immutable for provenance/audit only --
# current analyses use RAW_HOMOLOG_CANDIDATES_FASTA / CURATED_HOMOLOG_FASTA
# below instead. See data/provenance/SOURCES.md.
LEGACY_HOMOLOGS_DIR = DATA_DIR / "homologs"
LEGACY_RAW_HOMOLOG_FASTA = LEGACY_HOMOLOGS_DIR / "raw" / "flk_homologs_raw_aligned.fasta"

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
DOCS_DIR = REPO_ROOT / "docs"

FLK_SEQUENCE_JSON = SEQUENCE_DIR / "Q9SR13.json"
FLK_IUPRED_JSON = SEQUENCE_DIR / "Q9SR13_iupred2a.json"
FLK_PROVENANCE_JSON = PROVENANCE_DIR / "flk_sequence_provenance.json"

RAW_HOMOLOG_CANDIDATES_FASTA = RAW_DIR / "homolog_candidates_raw.fasta"
CURATED_HOMOLOG_FASTA = CURATED_DIR / "flk_homologs_curated.fasta"
HOMOLOG_PROVENANCE_TSV = PROVENANCE_DIR / "homolog_provenance.tsv"

for _d in (RESULTS_DIR, FIGURES_DIR, CURATED_DIR):
    _d.mkdir(parents=True, exist_ok=True)
