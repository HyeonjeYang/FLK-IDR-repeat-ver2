"""Network fetch (run once, cached): recover a chrysanthemum FLK coding
sequence from public transcriptome assemblies.

Huang et al. (2022, Plant Cell Environ 45:1442) characterized CmFLK
functionally, but no CmFLK accession could be found in NCBI Protein,
NCBI Nucleotide or UniProtKB, and the article is not open access. As a
documented substitute, A. thaliana FLK (Q9SR13) is searched with TBLASTN
against NCBI Transcriptome Shotgun Assembly contigs (tsa_nt) restricted to
the genus Chrysanthemum (txid13422); the best-hit contigs are downloaded
so their open reading frames can be translated offline
(scripts/build_expanded_candidates.py). Whether the recovered sequence is
identical to the clone of Huang et al. cannot be verified and is never
claimed.

Outputs:
- data/expanded/blast/chrysanthemum_tblastn.json
- data/expanded/chrysanthemum/<contig>.fasta (top contigs, nucleotide)
- appends a row to data/expanded/blast/blast_runs.tsv
"""
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_expanded_blast import BLAST_URL, OUT_DIR, fetch, wait  # noqa: E402
from flk_repeat.paths import DATA_DIR  # noqa: E402
from flk_repeat.sequence import load_flk_record  # noqa: E402

CONTIG_DIR = DATA_DIR / "expanded" / "chrysanthemum"
ENTREZ = "txid13422[ORGN]"
N_CONTIGS = 5


def main() -> None:
    out = OUT_DIR / "chrysanthemum_tblastn.json"
    if not out.exists():
        data = {"CMD": "Put", "PROGRAM": "tblastn", "DATABASE": "tsa_nt", "QUERY": load_flk_record()["sequence"],
                "ENTREZ_QUERY": ENTREZ, "HITLIST_SIZE": "50", "EXPECT": "1e-10", "MATRIX_NAME": "BLOSUM62",
                "TOOL": "flk-repeat-revision"}
        r = requests.post(BLAST_URL, data=data, timeout=120)
        r.raise_for_status()
        import re
        rid = re.search(r"RID = (\S+)", r.text).group(1)
        rtoe = int(re.search(r"RTOE = (\d+)", r.text).group(1))
        print("submitted", rid, flush=True)
        wait(rid, rtoe)
        out.write_text(fetch(rid), encoding="utf-8")
        with open(OUT_DIR / "blast_runs.tsv", "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh, delimiter="\t").writerow(
                ["chrysanthemum_tsa", rid, "Q9SR13", "tsa_nt (tblastn)", ENTREZ, "50", "1e-10", "BLOSUM62",
                 dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 out.relative_to(DATA_DIR.parent).as_posix()])

    report = json.loads(out.read_text(encoding="utf-8"))["BlastOutput2"][0]["report"]["results"]["search"]
    hits = report.get("hits", [])
    CONTIG_DIR.mkdir(parents=True, exist_ok=True)
    for hit in hits[:N_CONTIGS]:
        acc = hit["description"][0]["accession"]
        path = CONTIG_DIR / f"{acc}.fasta"
        if path.exists():
            continue
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                         params={"db": "nuccore", "id": acc, "rettype": "fasta", "retmode": "text",
                                 "tool": "flk-repeat-revision"}, timeout=120)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
        print("saved", acc, hit["description"][0].get("title", "")[:80], flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
