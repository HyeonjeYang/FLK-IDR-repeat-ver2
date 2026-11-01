"""Load and validate the audited A. thaliana FLK (Q9SR13) sequence record."""
import json

from .paths import FLK_SEQUENCE_JSON, FLK_IUPRED_JSON, FLK_PROVENANCE_JSON
from .provenance import sha256_of_sequence

EXPECTED_ACCESSION = "Q9SR13"
EXPECTED_LENGTH = 577
IUPRED_SEQUENCE_PREFIX = "IUPred2A:"


class ProvenanceError(ValueError):
    """Raised when a checked-in source file fails an identity/hash check."""


def load_provenance() -> dict:
    return json.loads(FLK_PROVENANCE_JSON.read_text(encoding="utf-8"))


def load_flk_record() -> dict:
    """Load the cached UniProtKB entry and validate it against the
    provenance record. Raises ProvenanceError on any mismatch instead of
    silently continuing.
    """
    prov = load_provenance()
    entry = json.loads(FLK_SEQUENCE_JSON.read_text(encoding="utf-8"))

    accession = entry["primaryAccession"]
    seq = entry["sequence"]["value"]
    length = entry["sequence"]["length"]
    reviewed = entry["entryType"].lower().startswith("uniprotkb reviewed")
    seq_hash = sha256_of_sequence(seq)

    if accession != prov["accession"] or accession != EXPECTED_ACCESSION:
        raise ProvenanceError(f"accession mismatch: {accession!r}")
    if length != len(seq) or length != prov["length_aa"] or length != EXPECTED_LENGTH:
        raise ProvenanceError(f"length mismatch: record={length} actual={len(seq)}")
    if seq_hash != prov["sequence_sha256_uppercase"]:
        raise ProvenanceError(f"sequence SHA256 mismatch: {seq_hash!r}")
    if reviewed != prov["reviewed"]:
        raise ProvenanceError(f"reviewed-status mismatch: entry={reviewed}")

    return {
        "accession": accession,
        "sequence": seq,
        "length": length,
        "reviewed": reviewed,
        "sha256": seq_hash,
        "provenance": prov,
    }


def load_iupred_scores(expected_length: int) -> list[float]:
    """Load cached IUPred2A 'long' disorder scores, validated against the
    embedded sequence string (after stripping the cache's own prefix).
    """
    cache = json.loads(FLK_IUPRED_JSON.read_text(encoding="utf-8"))
    embedded_seq = cache["sequence"]
    if embedded_seq.startswith(IUPRED_SEQUENCE_PREFIX):
        embedded_seq = embedded_seq[len(IUPRED_SEQUENCE_PREFIX):]
    scores = cache["iupred2"]
    if len(embedded_seq) != expected_length or len(scores) != expected_length:
        raise ProvenanceError(
            f"IUPred2A cache length mismatch: seq={len(embedded_seq)} "
            f"scores={len(scores)} expected={expected_length}"
        )
    if cache.get("source") != "iupred2a_api":
        raise ProvenanceError(f"IUPred2A cache source is not the real API: {cache.get('source')!r}")
    return scores
