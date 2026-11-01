# FLK N-terminal repeat analyses

Reproducible analyses for the LEPQQYEV repeat array and annotated FLK transcript models in *Arabidopsis thaliana*. The three exact protein copies were first described by [Mockler et al. (2004)](https://doi.org/10.1073/pnas.0404552101). This repository contains analysis code, versioned small sequence inputs, selection/QC tables, and derived results. It does not contain the manuscript, rendered figures, or large BLAST and proteome downloads.

## Run

Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python scripts/audit_flk_codon_splice.py
python scripts/analyze_revision_claims.py --section expanded
python scripts/analyze_revision_claims.py --section close
python scripts/analyze_nonoverlap_null.py --n 1000000
```

The first command audits four Col-0 transcript models, the 72-nt splice, repeat codons, and two external array examples. The next two regenerate the 26-species repeat summary and the close-relative comparisons from compact FASTA and TSV inputs. The final command performs independent global and 25-aa-block shuffles; with the published seed it takes longer and checks non-overlapping triplication. Outputs are written under `results/expanded/`. Run `python -m pytest -q` for code checks.

`data/expanded/nucleotide/SOURCES.md` lists accession-specific GenBank inputs. `data/expanded/blast/blast_runs.tsv` records the NCBI BLAST searches behind the expanded candidate selection; `data/expanded/sequences/representative_40.fasta` and `results/expanded/reciprocal_blast_audit.tsv` are the compact inputs for its repeat summary. Large BLAST JSON, GenPept intermediates, and the UniProt proteome stream are omitted. Their original collection parameters are recorded, but exact reruns of the upstream BLAST selection and IDR background extraction require those public-source downloads, which can change with database releases.

The 1-million-draw results and the other manuscript tables are included in `results/`. The short FLK splice is an **annotated transcript model**; RNA-seq junction support and short-protein expression have not been established here.
