#!/usr/bin/env python3
"""Background for the physicochemical-conservation analysis (revision):
amino-acid composition and descriptor distributions of disordered regions
in the A. thaliana reference proteome.

Source (cached, immutable; see data/expanded/SOURCES.md):
    data/expanded/background/athaliana_UP000006548_regions.tsv.gz
    UniProtKB REST stream, query proteome:UP000006548, fields accession,
    reviewed, length, ft_region, sequence.

Disordered regions = REGION features whose note is exactly "Disordered"
(MobiDB-lite consensus as propagated by UniProt). Identical region
sequences are counted once. Regions shorter than 30 residues are dropped
before the descriptor distribution is built, so that composition fractions
are not dominated by very short segments.

Writes:
    results/expanded/athaliana_idr_background_composition.tsv
    results/expanded/athaliana_idr_background_features.tsv
"""
import gzip
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat import evolution_sim as es  # noqa: E402
from flk_repeat import features as ft  # noqa: E402
from flk_repeat.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

SOURCE = DATA_DIR / "expanded" / "background" / "athaliana_UP000006548_regions.tsv.gz"
OUT_DIR = RESULTS_DIR / "expanded"
MIN_REGION_LENGTH = 30
REGION_RE = re.compile(r'REGION (\d+)\.\.(\d+); /note="([^"]*)"')


def disordered_regions() -> list[str]:
    seen = set()
    regions = []
    with gzip.open(SOURCE, "rt", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            acc, _reviewed, _length, region_field, seq = line.rstrip("\n").split("\t")
            for start, end, note in REGION_RE.findall(region_field):
                if note != "Disordered":
                    continue
                sub = seq[int(start) - 1:int(end)]
                if sub not in seen:
                    seen.add(sub)
                    regions.append(sub)
    return regions


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = disordered_regions()
    freqs = es.composition_vector(regions)
    comp = pd.DataFrame({"amino_acid": list(es.PAML_ORDER), "frequency": freqs})
    comp.to_csv(OUT_DIR / "athaliana_idr_background_composition.tsv", sep="\t", index=False)

    long_regions = [r for r in regions if len(ft.clean(r)) >= MIN_REGION_LENGTH]
    feats = pd.DataFrame([ft.compute_features(r, include_patterning=False) for r in long_regions])
    feats.to_csv(OUT_DIR / "athaliana_idr_background_features.tsv", sep="\t", index=False)
    print(f"A. thaliana disordered regions: {len(regions)} unique, "
          f"{len(long_regions)} >= {MIN_REGION_LENGTH} aa used for descriptor background")
    return {"freqs": freqs, "features": feats, "n_regions": len(regions)}


if __name__ == "__main__":
    main()
