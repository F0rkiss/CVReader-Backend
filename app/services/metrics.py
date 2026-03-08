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
    """
    n = len(ref)
    m = len(hyp)

    # Create distance matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[n][m]
