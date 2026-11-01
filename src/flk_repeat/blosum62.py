"""Standard BLOSUM62 substitution matrix (Henikoff & Henikoff, 1992),
embedded directly so this repository has no dependency on Biopython or
network access for a conservative-substitution score.
"""

_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

_MATRIX_ROWS = [
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]

BLOSUM62 = {
    (a, b): _MATRIX_ROWS[i][j]
    for i, a in enumerate(_AA_ORDER)
    for j, b in enumerate(_AA_ORDER)
}

GAP_SCORE = -4  # explicit, documented penalty for an aligned gap ('-') column


def score_pair(a: str, b: str) -> int:
    """BLOSUM62 score for one aligned residue pair; GAP_SCORE if either side
    is a gap ('-')."""
    if a == "-" or b == "-":
        return GAP_SCORE
    return BLOSUM62[(a, b)]


def alignment_score(query: str, reference: str) -> int:
    """Sum of per-position BLOSUM62 scores over two equal-length aligned
    strings (gaps allowed, scored via GAP_SCORE)."""
    if len(query) != len(reference):
        raise ValueError("alignment_score requires equal-length strings")
    return sum(score_pair(a, b) for a, b in zip(query, reference))


def self_score(reference: str) -> int:
    """Reference-against-itself score, used to normalize alignment_score
    into a [~0, 1] conservative-substitution similarity."""
    return sum(score_pair(a, a) for a in reference)


def normalized_similarity(query: str, reference: str) -> float:
    """alignment_score(query, reference) / self_score(reference), clipped at
    0. 1.0 means as similar as the reference is to itself; <=0 means no
    net conservative similarity signal."""
    denom = self_score(reference)
    return max(0.0, alignment_score(query, reference) / denom)
