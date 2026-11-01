"""Checks that the published result tables are populated and consistent."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flk_repeat.paths import RESULTS_DIR


@pytest.mark.parametrize("fname", [
    "flk_sequence_provenance.tsv",
    "flk_repeat_coordinates.tsv",
    "flk_window_composition.tsv",
    "flk_domain_annotation.tsv",
    "flk_homolog_repeat_comparison.tsv",
    "flk_null_model_summary.tsv",
    "flk_null_model_histogram.tsv",
])
def test_results_tsv_parses(fname):
    path = RESULTS_DIR / fname
    assert path.exists(), f"missing {path} -- run scripts/reproduce_all.py first"
    df = pd.read_csv(path, sep="\t")
    assert len(df) > 0


def test_flk_repeat_coordinates_are_exactly_lepqqyev():
    df = pd.read_csv(RESULTS_DIR / "flk_repeat_coordinates.tsv", sep="\t")
    assert len(df) == 3
    assert all(df.sequence == "LEPQQYEV")


def test_no_confirmed_ortholog_label_in_results():
    df = pd.read_csv(RESULTS_DIR / "flk_homolog_repeat_comparison.tsv", sep="\t")
    assert "CONFIRMED_ORTHOLOG" not in set(df.orthology_status)
    allowed = {"REFERENCE", "HIGH_CONFIDENCE_ORTHOLOG", "SUPPORTED_ORTHOLOG_CANDIDATE", "UNCERTAIN"}
    assert set(df.orthology_status) <= allowed


def test_no_forbidden_llps_or_physicochemical_class_language():
    """This project must never claim LLPS/condensate formation as observed,
    and must never reuse the old, chemically-inaccurate 'E/Q/Y
    physicochemical class' terminology as an actual label/column anywhere
    in generated outputs or code. (Docs may still explain, in prose, why
    the phrase is avoided -- that is a negation, not a usage, and is
    reviewed by hand rather than by this regex-based check.)"""
    forbidden_class = re.compile(r"physicochemical class", re.IGNORECASE)
    forbidden_llps_claims = re.compile(
        r"FLK (undergoes|shows|exhibits|drives|is (shown|known|proven) to (undergo|drive))\b"
        r"|LEPQQYEV drives|drives (LLPS|phase separation)"
        r"|(experimentally )?proven functional motif",
        re.IGNORECASE)

    code_files = list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").glob("*.py"))
    result_files = list(RESULTS_DIR.glob("*.tsv"))

    llps_offenders, class_offenders = [], []
    for f in code_files:
        text = f.read_text(encoding="utf-8")
        for line in text.splitlines():
            if forbidden_llps_claims.search(line):
                llps_offenders.append((str(f), line.strip()))
            if forbidden_class.search(line):
                class_offenders.append((str(f), line.strip()))
    for f in result_files:
        text = f.read_text(encoding="utf-8")
        if forbidden_class.search(text):
            class_offenders.append((str(f), "(found in generated results table)"))

    assert not class_offenders, f"found 'E/Q/Y physicochemical class' language: {class_offenders}"
    assert not llps_offenders, f"found an unguarded LLPS/condensate/proven-motif claim: {llps_offenders}"


def test_no_absolute_local_paths_in_source():
    forbidden = re.compile(r"[A-Za-z]:[\\/]Users[\\/]|/home/|/Users/", re.IGNORECASE)
    offenders = []
    for f in list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").glob("*.py")):
        text = f.read_text(encoding="utf-8")
        if forbidden.search(text):
            offenders.append(str(f))
    assert not offenders, f"found absolute local paths in: {offenders}"
