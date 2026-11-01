#!/usr/bin/env python3
"""Choose the species of the expanded FLK sampling from the cached BLAST
hits (offline; revision, Reviewer 1 comment 1).

Selection is a priori: the target list below was fixed before any
N-terminal sequence was inspected. It names one species per major
angiosperm lineage that has a RefSeq-annotated genome, favouring crops and
model species, and it always includes species with experimental data on
an FLK homolog (sugar beet; chrysanthemum through its TSA contig, handled
separately; rice is already in the curated set). Each target lists
fallbacks from the same order, used in the given order only if the first
choice has no hit. Species already in the 22-sequence curated set are not
re-selected.

For every selected species, all hits with E <= 1e-20 and query coverage
>= 0.5 (FLK and its KH-domain paralogs) are kept for the gene-family tree.

Writes:
    data/expanded/selection/selected_species.tsv
    data/expanded/selection/candidate_hits.tsv
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat.expanded import EXPANDED_DATA, load_all_blast_hits  # noqa: E402

OUT_DIR = EXPANDED_DATA / "selection"
MAX_EVALUE = 1e-20
MIN_QUERY_COVERAGE = 0.5

# (lineage, order, [first choice, fallbacks...], reason)
TARGETS = [
    ("ANA grade", "Amborellales", ["Amborella trichopoda"], "sister to all other angiosperms"),
    ("ANA grade", "Nymphaeales", ["Nymphaea colorata", "Nymphaea thermarum"], "water lilies"),
    ("Magnoliids", "Laurales", ["Cinnamomum micranthum f. kanehirae", "Persea americana", "Cinnamomum kanehirae"], "magnoliid"),
    ("Magnoliids", "Magnoliales", ["Magnolia sinica", "Liriodendron chinense"], "magnoliid"),
    ("Monocots", "Alismatales", ["Spirodela polyrhiza", "Spirodela intermedia", "Lemna minor", "Colocasia esculenta"], "early-diverging monocot"),
    ("Monocots", "Dioscoreales", ["Dioscorea cayenensis subsp. rotundata", "Dioscorea alata"], "yam"),
    ("Monocots", "Asparagales", ["Asparagus officinalis"], "Asparagaceae"),
    ("Monocots", "Asparagales", ["Phalaenopsis equestris", "Dendrobium catenatum", "Apostasia shenzhenica"], "orchid"),
    ("Monocots", "Arecales", ["Elaeis guineensis", "Phoenix dactylifera", "Cocos nucifera"], "palm"),
    ("Monocots", "Zingiberales", ["Musa acuminata AAA Group", "Musa acuminata subsp. malaccensis", "Zingiber officinale"], "banana"),
    ("Monocots", "Poales", ["Ananas comosus"], "Bromeliaceae"),
    ("Monocots", "Poales", ["Brachypodium distachyon"], "Poaceae, Pooideae model"),
    ("Monocots", "Poales", ["Hordeum vulgare subsp. vulgare", "Hordeum vulgare"], "Poaceae, barley"),
    ("Monocots", "Poales", ["Triticum aestivum"], "Poaceae, wheat"),
    ("Monocots", "Poales", ["Zea mays"], "Poaceae, maize"),
    ("Monocots", "Poales", ["Sorghum bicolor"], "Poaceae, sorghum"),
    ("Monocots", "Poales", ["Setaria italica", "Setaria viridis"], "Poaceae, foxtail millet"),
    ("Early-diverging eudicots", "Ranunculales", ["Papaver somniferum", "Aquilegia coerulea", "Macleaya cordata"], "Ranunculales"),
    ("Early-diverging eudicots", "Proteales", ["Nelumbo nucifera"], "lotus"),
    ("Caryophyllales", "Caryophyllales", ["Beta vulgaris subsp. vulgaris", "Beta vulgaris"], "functional data: BvFLK complements flk"),
    ("Caryophyllales", "Caryophyllales", ["Chenopodium quinoa"], "second Amaranthaceae"),
    ("Superrosids", "Vitales", ["Vitis vinifera"], "grapevine"),
    ("Rosids (fabids)", "Fabales", ["Glycine max"], "soybean"),
    ("Rosids (fabids)", "Fabales", ["Medicago truncatula"], "Medicago"),
    ("Rosids (fabids)", "Cucurbitales", ["Cucumis sativus"], "cucumber"),
    ("Rosids (fabids)", "Malpighiales", ["Populus trichocarpa"], "poplar"),
    ("Rosids (fabids)", "Fagales", ["Juglans regia", "Quercus suber"], "walnut"),
    ("Rosids (fabids)", "Rosales", ["Malus domestica", "Fragaria vesca subsp. vesca"], "apple"),
    ("Rosids (malvids)", "Malvales", ["Theobroma cacao"], "cacao"),
    ("Rosids (malvids)", "Malvales", ["Gossypium raimondii", "Gossypium hirsutum"], "cotton"),
    ("Rosids (malvids)", "Sapindales", ["Citrus sinensis", "Citrus clementina"], "citrus"),
    ("Rosids (malvids)", "Myrtales", ["Eucalyptus grandis"], "eucalyptus"),
    ("Rosids (malvids)", "Brassicales", ["Carica papaya"], "Caricaceae, Brassicales outgroup"),
    ("Rosids (malvids)", "Brassicales", ["Tarenaya hassleriana"], "Cleomaceae, sister family of Brassicaceae"),
    ("Asterids", "Solanales", ["Solanum lycopersicum"], "tomato"),
    ("Asterids", "Solanales", ["Nicotiana tabacum", "Nicotiana sylvestris"], "tobacco"),
    ("Asterids", "Gentianales", ["Coffea arabica", "Coffea eugenioides"], "coffee"),
    ("Asterids", "Lamiales", ["Olea europaea var. sylvestris", "Olea europaea subsp. europaea"], "olive"),
    ("Asterids", "Apiales", ["Daucus carota subsp. sativus"], "carrot"),
    ("Asterids", "Asterales", ["Helianthus annuus"], "Asteraceae, sunflower"),
    ("Asterids", "Asterales", ["Lactuca sativa"], "Asteraceae, lettuce"),
    ("Brassicaceae", "Brassicales", ["Aethionema arabicum"], "earliest-diverging Brassicaceae lineage"),
    ("Brassicaceae", "Brassicales", ["Schrenkiella parvula", "Eutrema parvulum"], "lineage II"),
    ("Brassicaceae", "Brassicales", ["Brassica oleracea var. oleracea", "Brassica oleracea"], "Brassiceae, C genome"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hits = pd.DataFrame(load_all_blast_hits())
    good = hits[(hits.evalue <= MAX_EVALUE) & (hits.query_coverage >= MIN_QUERY_COVERAGE)]
    present = set(good.sciname)

    rows, chosen = [], []
    for lineage, order, names, reason in TARGETS:
        pick = next((n for n in names if n in present), None)
        rows.append({"lineage": lineage, "order": order, "first_choice": names[0],
                     "selected_species": pick or "", "used_fallback": bool(pick and pick != names[0]),
                     "reason": reason,
                     "n_kh_family_hits": int((good.sciname == pick).sum()) if pick else 0})
        if pick:
            chosen.append(pick)
    sel = pd.DataFrame(rows)
    sel.to_csv(OUT_DIR / "selected_species.tsv", sep="\t", index=False)

    cand = good[good.sciname.isin(chosen)].sort_values(["sciname", "bit_score"], ascending=[True, False])
    cand.to_csv(OUT_DIR / "candidate_hits.tsv", sep="\t", index=False)
    missing = sel[sel.selected_species == ""]
    print(f"selected {len(chosen)} species; {len(missing)} targets without any hit: "
          f"{', '.join(missing.first_choice)}")
    print(f"{len(cand)} KH-family hits kept for the selected species")


if __name__ == "__main__":
    main()
