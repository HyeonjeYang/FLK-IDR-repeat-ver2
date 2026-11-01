# Expanded analysis inputs

`blast/blast_runs.tsv` gives the NCBI RefSeq BLASTP request IDs, taxonomic filters, hit limits, and retrieval dates. The large JSON responses are omitted from this compact release. NCBI request IDs can expire; a new search can differ as RefSeq changes.

`selection/candidate_hits.tsv` preserves the forward-hit selection table. `sequences/representative_40.fasta` contains the selected proteins, and `results/expanded/reciprocal_blast_audit.tsv` preserves reciprocal-search and repeat-region QC decisions. Together they let `scripts/analyze_revision_claims.py --section expanded` recompute the 26-species repeat description without large BLAST or GenPept files.

`nucleotide/SOURCES.md` lists the versioned NCBI records used for the transcript and codon audit. `results/expanded/athaliana_idr_150aa_window_repeats.tsv` retains the window-level IDR background results. The original UniProt reference-proteome stream (UP000006548) and the full feature table are omitted; regenerating that background requires a new public-source download and may reflect a newer release.
