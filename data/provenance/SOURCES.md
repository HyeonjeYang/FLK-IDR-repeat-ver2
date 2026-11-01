# Source material and its provenance

This repository is self-contained: every file under `data/` needed to
regenerate `results/` and `figures/` is checked into this repository, and
`scripts/reproduce_all.py` runs offline. This file records where each
checked-in source file originally came from.

## FLK primary sequence and disorder cache

- `data/sequence/Q9SR13.json` — cached UniProtKB entry for Q9SR13 (FLK,
  *Arabidopsis thaliana*). Original source: UniProtKB REST API
  (`rest.uniprot.org`).
- `data/sequence/Q9SR13_iupred2a.json` — cached IUPred2A ("long" disorder
  type) per-residue scores for the same sequence. Original source: IUPred2A
  REST API (`iupred2a.elte.hu`).
- Both files were validated on import: the SHA256 of the uppercase sequence
  string in `Q9SR13.json` was recomputed independently and matches
  `data/provenance/flk_sequence_provenance.json`. The IUPred2A cache's
  embedded sequence (after stripping its `"IUPred2A:"` prefix) was
  confirmed character-identical to the UniProt sequence, with exactly 577
  per-residue scores.
- These two files were originally copied from a prior, read-only reference
  repository (`HyeonjeYang/arabidopsis-rbp-idr-atlas`), which had itself
  retrieved and validated them from the APIs above.

## FLK homolog/ortholog-candidate set (current, expanded)

- `data/raw/homolog_candidates_raw.fasta` — 32 candidate sequences,
  independently retrieved from the UniProtKB REST API
  (`rest.uniprot.org/uniprotkb/<ACCESSION>.json`) on **2026-08-26**.
  Immutable: never edited, trimmed, or reordered by any script.
- `data/provenance/homolog_provenance.tsv` — one row per candidate above,
  recording species, accession, database, length, SHA256, orthology-
  confidence tier, KH-domain-annotation status, N-terminal-completeness
  status, inclusion decision, and the specific reason for that decision.
  22 of the 32 candidates are included in the curated comparison; the
  other 10 are retained in both files with an explicit exclusion reason
  (redundant subgenome paralog, synthetic lab-cross hybrid, duplicate
  accession for the same locus, or an alternate isoform of the reference
  species itself) — never silently dropped.
- `data/curated/flk_homologs_curated.fasta` — the 22 included sequences
  only, ungapped, generated from the two files above by
  `scripts/analyze_repeat_conservation.py` (gap removal / filtering only,
  no sequence edits). `scripts/reproduce_all.py` checks this file is
  byte-identical when regenerated.
- **Discovery methodology** (full detail in `docs/METHODS.md`): candidates
  were found via UniProt's automated protein-name propagation
  ("Flowering locus K homology domain", assigned by UniProt's rule-based
  annotation to sequences with strong similarity to a reviewed reference)
  restricted by taxonomy, then filtered using each candidate's own
  UniProt-annotated KH-domain architecture (FLK has exactly 3 KH domains)
  and, as an additional homology check, an exact/partial match to a
  conserved 31-aa sequence immediately downstream of FLK's own KH1 domain.
  No BLAST hit was treated as sufficient evidence alone.
- All 22 included sequences carry independently UniProt-annotated KH
  domains except *A. halleri* (A0ACE2D236), whose proteome carries no
  UniProt feature annotation at all; for that one record, the KH1 anchor
  position used for repeat-mapping is estimated from the 31-aa fingerprint
  match instead (documented in `homolog_provenance.tsv`,
  `kh_domain_status=fingerprint_match_no_feature_annotation`).

## FLK homolog alignment (legacy, superseded)

- `data/homologs/raw/flk_homologs_raw_aligned.fasta` — an 8-record
  Clustal-Omega alignment copied verbatim, byte-for-byte, from a prior,
  read-only reference repository (`HyeonjeYang/arabidopsis-rbp-idr-atlas`,
  `data/flk/flk_brassicaceae_clustalo_alignment.fasta`). Per that
  repository's own documentation, it was built from a legacy, taxonomically
  unscreened 33-sequence BLAST-hit set of which only these 8 records
  survived into the cached file; the pre-filter set and most accessions are
  not recoverable.
- **This file is retained immutable for historical/audit provenance only.**
  It is no longer the input to any current script — the expanded, curated
  set above (`data/raw/`, `data/curated/`,
  `data/provenance/homolog_provenance.tsv`) is used for all current
  analyses and figures. Three of its four non-reference species (*A.
  lyrata*, *A. halleri*, *Arabis alpina*) were independently re-identified
  with real UniProtKB accessions during the expansion (D7L277, A0ACE2D236,
  A0A087H6X3) and were confirmed to carry byte-identical sequence content
  to the legacy alignment's rows — an independent cross-validation of the
  prior repository's own curation, not a correction of it.

## Rice ("putative FLK ortholog") — literature context, `UNCERTAIN` tier

See `docs/LITERATURE_CONTEXT.md`. A modern candidate (UniProtKB Q10GH8,
*Oryza sativa* japonica, locus Os03g0627500) was independently identified
via the same KH-domain-architecture search and is included in
`homolog_provenance.tsv`/the comparison table with `orthology_status=
UNCERTAIN` — not `CONFIRMED_ORTHOLOG` and not treated as proven identical
to either 2004 paper's own analyzed sequence.

## What was intentionally excluded from this repository

This is a FLK-only project. The original read-only reference repository
also contains ECT2/3/4/8, ALKBH9B/9C/10B, FCA, FPA, FLC sequences and
legacy LLPS-predictor material (FuzDrop/catGRANULE/PScore transcripts, an
ECT8 legacy-accession contrast, an ALKBH deletion "synergy" audit
correction). None of that is copied here; it is out of scope for the
triplicated LEPQQYEV question this repository addresses.
