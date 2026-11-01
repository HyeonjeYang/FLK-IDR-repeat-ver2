"""Network fetch (run once, outputs cached): sequences and verification
records for the expanded FLK candidates (revision, Reviewer 1 comment 1).

Stages (each skipped when its output already exists):
  1. sequences  - protein FASTA and GenPept records (for the GeneID that
                  groups RefSeq isoforms of one gene) of every KH-family hit
                  of the selected species (data/expanded/selection/
                  candidate_hits.tsv), plus the A. thaliana KH paralogs.
  2. reverse    - one BLASTP of every representative against A. thaliana
                  RefSeq proteins (txid3702), for the reciprocal-best-hit
                  check; representatives come from
                  flk_repeat.expanded.choose_representatives.
     reverse_subjects - GenPept of the A. thaliana subjects of that search,
                  for their TAIR locus tags.
  3. uniparc    - UniParc entry (by MD5 of the exact sequence) and all of
                  its database cross-references, for every representative
                  and for the 22 curated sequences.

No e-mail address or other personal identifier is sent to any service.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_expanded_blast import BLAST_URL, fetch, wait  # noqa: E402
from flk_repeat import expanded as ex  # noqa: E402
from flk_repeat.homologs import parse_fasta, parse_raw_candidates  # noqa: E402
from flk_repeat.paths import RAW_HOMOLOG_CANDIDATES_FASTA, HOMOLOG_PROVENANCE_TSV  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SEQ_DIR = ex.EXPANDED_DATA / "sequences"
VERIFY_DIR = ex.EXPANDED_DATA / "verification"
TOOL = "flk-repeat-revision"


def efetch(ids: list[str], rettype: str) -> str:
    out = []
    for i in range(0, len(ids), 100):
        r = requests.post(EUTILS, data={"db": "protein", "id": ",".join(ids[i:i + 100]),
                                        "rettype": rettype, "retmode": "text", "tool": TOOL}, timeout=300)
        r.raise_for_status()
        out.append(r.text)
        time.sleep(0.5)
    return "".join(out)


def stage_sequences() -> None:
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    fasta_path, gp_path = SEQ_DIR / "candidate_hits.fasta", SEQ_DIR / "candidate_hits.gp"
    if fasta_path.exists() and gp_path.exists():
        print("sequences: cached")
        return
    ids = pd.read_csv(ex.EXPANDED_DATA / "selection" / "candidate_hits.tsv", sep="\t").accession.tolist()
    fasta_path.write_text(efetch(ids, "fasta"), encoding="utf-8")
    gp_path.write_text(efetch(ids, "gp"), encoding="utf-8")
    print(f"sequences: saved {len(ids)} FASTA + GenPept records")


def stage_isoforms() -> None:
    """Fetch every RefSeq isoform of each representative's gene (the BLAST
    hit list can omit isoforms that score lower)."""
    path = SEQ_DIR / "representative_gene_isoforms.gp"
    if path.exists():
        print("isoforms: cached")
        return
    reps = ex.choose_representatives(isoform_records=None)
    gene_ids = sorted({r["gene_id"] for r in reps if r["gene_id"]})
    acc = []
    for gid in gene_ids:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
                         params={"dbfrom": "gene", "db": "protein", "id": gid, "linkname": "gene_protein_refseq",
                                 "retmode": "json", "tool": TOOL}, timeout=120)
        r.raise_for_status()
        for linkset in r.json().get("linksets", []):
            for db in linkset.get("linksetdbs", []):
                acc.extend(db.get("links", []))
        time.sleep(0.4)
    path.write_text(efetch([str(a) for a in acc], "gp"), encoding="utf-8")
    print(f"isoforms: saved {len(acc)} GenPept records for {len(gene_ids)} genes")


def stage_reverse() -> None:
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    out = VERIFY_DIR / "reverse_blast_athaliana.json"
    if out.exists():
        print("reverse BLAST: cached")
        return
    pending = VERIFY_DIR / "reverse_blast.pending.json"
    if pending.exists():
        job = json.loads(pending.read_text(encoding="utf-8"))
        rid, rtoe = job["rid"], 0
        print("reverse BLAST resuming", rid, flush=True)
    else:
        reps = ex.choose_representatives()
        query = "".join(f">{r['accession']}\n{r['sequence']}\n" for r in reps)
        data = {"CMD": "Put", "PROGRAM": "blastp", "DATABASE": "refseq_protein", "QUERY": query,
                "ENTREZ_QUERY": "txid3702[ORGN]", "HITLIST_SIZE": "10", "EXPECT": "1e-5",
                "MATRIX_NAME": "BLOSUM62", "TOOL": TOOL}
        r = requests.post(BLAST_URL, data=data, timeout=300)
        r.raise_for_status()
        import re
        rid = re.search(r"RID = (\S+)", r.text).group(1)
        rtoe = int(re.search(r"RTOE = (\d+)", r.text).group(1))
        pending.write_text(json.dumps({"rid": rid, "rtoe": rtoe,
                                       "query_accessions": [r["accession"] for r in reps]},
                                      indent=2) + "\n", encoding="utf-8")
        print("reverse BLAST submitted", rid, flush=True)
    wait(rid, rtoe)
    out.write_text(fetch(rid), encoding="utf-8")
    (VERIFY_DIR / "reverse_blast_rid.txt").write_text(rid + "\n", encoding="utf-8")
    pending.unlink()
    print("reverse BLAST: saved")


def stage_reverse_subjects() -> None:
    """GenPept of every A. thaliana subject in the reverse search, for its
    locus tag (AT3G04610 = FLK)."""
    out = VERIFY_DIR / "reverse_blast_athaliana_subjects.gp"
    if out.exists():
        print("reverse subjects: cached")
        return
    doc = json.loads((VERIFY_DIR / "reverse_blast_athaliana.json").read_text(encoding="utf-8"))
    acc = sorted({h["description"][0]["accession"]
                  for rep in doc["BlastOutput2"] for h in rep["report"]["results"]["search"].get("hits", [])})
    out.write_text(efetch(acc, "gp"), encoding="utf-8")
    print(f"reverse subjects: saved {len(acc)} GenPept records")


def uniparc_record(seq: str) -> dict:
    md5 = hashlib.md5(seq.encode()).hexdigest().upper()
    r = requests.get("https://rest.uniprot.org/uniparc/search",
                     params={"query": f"checksum:{md5}", "format": "json", "size": 5}, timeout=120)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return {"md5": md5, "upi": None, "cross_references": []}
    upi = results[0]["uniParcId"]
    r = requests.get(f"https://rest.uniprot.org/uniparc/{upi}/databases",
                     params={"format": "json", "size": 500}, timeout=120)
    r.raise_for_status()
    xrefs = [{"database": x.get("database"), "id": x.get("id"), "active": x.get("active"),
              "taxon_id": (x.get("organism") or {}).get("taxonId")} for x in r.json().get("results", [])]
    return {"md5": md5, "upi": upi, "sequence_length": len(seq), "cross_references": xrefs}


def stage_uniparc() -> None:
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    out = VERIFY_DIR / "uniparc_crossrefs.json"
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    seqs = {r["accession"]: r["sequence"] for r in ex.choose_representatives()}
    raw = parse_raw_candidates(RAW_HOMOLOG_CANDIDATES_FASTA)
    prov = pd.read_csv(HOMOLOG_PROVENANCE_TSV, sep="\t")
    seqs.update({a: raw[a] for a in prov[prov.included].accession})
    for acc, seq in seqs.items():
        if acc in done:
            continue
        done[acc] = uniparc_record(seq)
        time.sleep(0.3)
        out.write_text(json.dumps(done, indent=1), encoding="utf-8")
    print(f"UniParc: {len(done)} records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*",
                    default=["sequences", "isoforms", "reverse", "reverse_subjects", "uniparc"])
    args = ap.parse_args()
    for stage in args.stages:
        {"sequences": stage_sequences, "isoforms": stage_isoforms, "reverse": stage_reverse,
         "reverse_subjects": stage_reverse_subjects, "uniparc": stage_uniparc}[stage]()


if __name__ == "__main__":
    main()
