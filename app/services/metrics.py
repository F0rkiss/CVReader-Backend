try:
    from rapidfuzz.distance import Levenshtein as _rf_levenshtein
except ModuleNotFoundError:  # pragma: no cover
    _rf_levenshtein = None


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).

    CER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions
    N = total characters in reference

    Lower is better. 0.0 = perfect match.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    ref = list(reference)
    hyp = list(hypothesis)

    # Edit distance (Levenshtein) at character level
    d = _levenshtein_distance(ref, hyp)

    return round(d / len(ref), 4)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions
    N = total words in reference

    Lower is better. 0.0 = perfect match.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    ref = reference.split()
    hyp = hypothesis.split()

    # Edit distance at word level
    d = _levenshtein_distance(ref, hyp)

    return round(d / len(ref), 4)


def _levenshtein_distance(ref: list, hyp: list) -> int:
    """
    Calculate Levenshtein edit distance between two sequences.
    Works for both character lists and word lists.

    Uses rapidfuzz (C implementation) when available, otherwise falls back
    to the pure-Python version below.
    """
    if _rf_levenshtein is not None:
        return int(_rf_levenshtein.distance(ref, hyp))
    return _levenshtein_distance_py(ref, hyp)


def _levenshtein_distance_py(ref: list, hyp: list) -> int:
    """
    Pure-Python Levenshtein edit distance.

    Keeps only two rows of the DP matrix, so memory is O(len(hyp))
    instead of O(len(ref) * len(hyp)).
    """
    m = len(hyp)

    prev = list(range(m + 1))
    for i in range(1, len(ref) + 1):
        curr = [i] + [0] * m
        ref_item = ref[i - 1]
        for j in range(1, m + 1):
            if ref_item == hyp[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(
                    prev[j],      # deletion
                    curr[j - 1],  # insertion
                    prev[j - 1],  # substitution
                )
        prev = curr

    return prev[m]
