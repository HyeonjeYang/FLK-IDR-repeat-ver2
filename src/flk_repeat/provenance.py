"""Sequence hashing helpers, used consistently for every provenance check."""
import hashlib


def sha256_of_sequence(seq: str) -> str:
    """SHA256 of the uppercase amino-acid string (whitespace-sensitive)."""
    return hashlib.sha256(seq.strip().upper().encode("utf-8")).hexdigest()


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
