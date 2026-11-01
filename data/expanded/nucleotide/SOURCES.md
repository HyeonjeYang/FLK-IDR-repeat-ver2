# FLK transcript and nucleotide records

These versioned GenBank files were retrieved from NCBI E-utilities on 1 November 2026 with `rettype=gbwithparts&retmode=text`. They preserve the annotated exon and CDS features used by `scripts/audit_flk_codon_splice.py`.

| File | NCBI database | Request ID |
| --- | --- | --- |
| `AY070475.1.gb` | nuccore | `AY070475.1` |
| `NM_111333.5.gb` | nuccore | `NM_111333.5` |
| `NM_001337535.1.gb` | nuccore | `NM_001337535.1` |
| `NM_001337536.1.gb` | nuccore | `NM_001337536.1` |
| `NM_001337537.1.gb` | nuccore | `NM_001337537.1` |
| `NC_003074.8_1250553_1254984.gb` | nuccore | `NC_003074.8&seq_start=1250553&seq_stop=1254984` |
| `EFH60713.1.gb` | protein | `EFH60713.1` |
| `GL348715.1_1584774_1588296.gb` | nuccore | `GL348715.1&seq_start=1584774&seq_stop=1588296` |

Endpoint: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=AY070475.1&rettype=gbwithparts&retmode=text` (replace `db`, `id`, and optional sequence bounds as listed above).

The Col-0 short transcript models annotate a 72-nt GT-AG splice inside the first coding exon. This is annotation-level evidence; locus-specific junction reads and translation of the short product were not assessed.
