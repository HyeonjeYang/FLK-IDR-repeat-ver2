"""Scientific-name labels for Matplotlib figures.

Italicize genus, species epithet, and infraspecific epithet; keep rank
abbreviations and cultivar/qualifier text upright. These strings are display
labels only and never alter the provenance names in result tables.
"""


def scientific_name(name: str) -> str:
    parts = name.split()
    if len(parts) < 2:
        return name
    genus, epithet, *rest = parts
    label = rf"$\mathit{{{genus}\ {epithet}}}$"
    if not rest:
        return label
    if len(rest) >= 2 and rest[0] in {"subsp.", "var.", "f."}:
        label += f" {rest[0]} " + rf"$\mathit{{{rest[1]}}}$"
        rest = rest[2:]
    return label + (" " + " ".join(rest) if rest else "")
