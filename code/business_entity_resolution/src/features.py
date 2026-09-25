"""
Feature engineering module computing multi-attribute similarity features
between Source 1 and candidate entity pairs.

Feature set (28 dimensions):
  Name features (12):
    - Fuzzy ratio, token-sort ratio, token-set ratio, partial ratio
    - Token Jaccard (word-level), character 3-gram Jaccard
    - Clean exact match, 5-char prefix match
    - Domain bidirectional match score
    - Normalized length difference
    - Name bigram Jaccard
    - Suffix-stripped name token-set ratio

  Address features (14):
    - Empty address flag
    - Fuzzy ratio, token-sort ratio, token-set ratio
    - Token Jaccard, character 3-gram Jaccard
    - Street number overlap ratio, any number match flag
    - Number count mismatch flag
    - Address locality token Jaccard (>=5 char tokens)
    - Address length difference
    - Both addresses empty flag

  Source indicator (2):
    - is_source2, is_source3
"""

from typing import List, Set
from rapidfuzz import fuzz

try:
    from .preprocessing import (
        normalize_text,
        clean_alphanumeric,
        extract_numbers,
        extract_domain_core,
        normalize_business_name,
        extract_name_tokens,
        extract_address_tokens,
    )
except ImportError:
    from preprocessing import (
        normalize_text,
        clean_alphanumeric,
        extract_numbers,
        extract_domain_core,
        normalize_business_name,
        extract_name_tokens,
        extract_address_tokens,
    )


FEATURE_NAMES = [
    # Name features
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard_tokens",
    "name_char_trigram_jaccard",
    "name_clean_exact",
    "name_prefix_match",
    "name_domain_match",
    "name_len_diff",
    "name_bigram_jaccard",
    "name_suffix_stripped_token_set",
    # Address features
    "addr_is_empty",
    "addr_both_empty",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard_tokens",
    "addr_char_trigram_jaccard",
    "addr_num_overlap_ratio",
    "addr_any_num_match",
    "addr_num_count_mismatch",
    "addr_locality_jaccard",
    "addr_len_diff",
    # Source indicator
    "cand_is_source2",
    "cand_is_source3",
    # Combined
    "name_x_addr_score",
    "blocking_score_proxy",
]


def _char_ngram_set(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text (no spaces)."""
    t = text.replace(" ", "")
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    cand_id: str,
    cand_name: str,
    cand_addr: str
) -> List[float]:
    """Compute a 28-dimensional feature vector for a candidate pair."""

    # ---------------------------------------------------------------
    # Pre-normalized representations
    # ---------------------------------------------------------------
    s1_n_norm = normalize_text(s1_name)
    cand_n_norm = normalize_text(cand_name)
    s1_a_norm = normalize_text(s1_addr) if s1_addr else ""
    cand_a_norm = normalize_text(cand_addr) if cand_addr else ""

    s1_n_clean = clean_alphanumeric(s1_name)
    cand_n_clean = clean_alphanumeric(cand_name)

    # Suffix-stripped (legal suffixes removed)
    s1_n_tokens_list = extract_name_tokens(s1_name)
    cand_n_tokens_list = extract_name_tokens(cand_name)
    s1_n_stripped = " ".join(s1_n_tokens_list)
    cand_n_stripped = " ".join(cand_n_tokens_list)

    # ---------------------------------------------------------------
    # 1. Name fuzzy similarities
    # ---------------------------------------------------------------
    name_ratio = fuzz.ratio(s1_n_norm, cand_n_norm) / 100.0
    name_token_sort_ratio = fuzz.token_sort_ratio(s1_n_norm, cand_n_norm) / 100.0
    name_token_set_ratio = fuzz.token_set_ratio(s1_n_norm, cand_n_norm) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_n_norm, cand_n_norm) / 100.0

    # 2. Name token Jaccard (word-level)
    s1_n_tokens = set(s1_n_norm.split())
    cand_n_tokens = set(cand_n_norm.split())
    name_jaccard_tokens = _jaccard(s1_n_tokens, cand_n_tokens)

    # 3. Name character trigram Jaccard
    s1_n_trig = _char_ngram_set(s1_n_clean, 3)
    cand_n_trig = _char_ngram_set(cand_n_clean, 3)
    name_char_trigram_jaccard = _jaccard(s1_n_trig, cand_n_trig)

    # 4. Exact clean match and 5-char prefix match
    name_clean_exact = 1.0 if (s1_n_clean and s1_n_clean == cand_n_clean) else 0.0
    if len(s1_n_clean) >= 5 and len(cand_n_clean) >= 5:
        name_prefix_match = 1.0 if s1_n_clean[:5] == cand_n_clean[:5] else 0.0
    else:
        name_prefix_match = 0.0

    # 5. Domain bidirectional match
    s1_dom = extract_domain_core(s1_name)
    cand_dom = extract_domain_core(cand_name)
    name_domain_match = 0.0
    if cand_dom and s1_n_clean and (cand_dom in s1_n_clean or s1_n_clean in cand_dom):
        name_domain_match = 1.0
    elif s1_dom and cand_n_clean and (s1_dom in cand_n_clean or cand_n_clean in s1_dom):
        name_domain_match = 1.0
    # Partial domain match via clean alphanumeric probe
    if name_domain_match == 0.0:
        if cand_dom and s1_n_clean and cand_dom[:8] == s1_n_clean[:8]:
            name_domain_match = 0.8
        elif s1_dom and cand_n_clean and s1_dom[:8] == cand_n_clean[:8]:
            name_domain_match = 0.8

    # 6. Name length difference (normalized)
    max_name_len = max(len(s1_name), len(cand_name), 1)
    name_len_diff = abs(len(s1_name) - len(cand_name)) / max_name_len

    # 7. Name bigram Jaccard
    s1_n_bigram = _char_ngram_set(s1_n_clean, 2)
    cand_n_bigram = _char_ngram_set(cand_n_clean, 2)
    name_bigram_jaccard = _jaccard(s1_n_bigram, cand_n_bigram)

    # 8. Suffix-stripped token-set ratio (legal suffixes removed)
    name_suffix_stripped_token_set = fuzz.token_set_ratio(s1_n_stripped, cand_n_stripped) / 100.0

    # ---------------------------------------------------------------
    # Address features
    # ---------------------------------------------------------------
    addr_is_empty = 1.0 if not cand_a_norm else 0.0
    addr_both_empty = 1.0 if (not s1_a_norm and not cand_a_norm) else 0.0

    if s1_a_norm and cand_a_norm:
        addr_ratio = fuzz.ratio(s1_a_norm, cand_a_norm) / 100.0
        addr_token_sort_ratio = fuzz.token_sort_ratio(s1_a_norm, cand_a_norm) / 100.0
        addr_token_set_ratio = fuzz.token_set_ratio(s1_a_norm, cand_a_norm) / 100.0

        s1_a_tokens = set(s1_a_norm.split())
        cand_a_tokens = set(cand_a_norm.split())
        addr_jaccard_tokens = _jaccard(s1_a_tokens, cand_a_tokens)

        # Character trigram Jaccard for addresses
        s1_a_clean = clean_alphanumeric(s1_addr)
        cand_a_clean = clean_alphanumeric(cand_addr)
        s1_a_trig = _char_ngram_set(s1_a_clean, 3)
        cand_a_trig = _char_ngram_set(cand_a_clean, 3)
        addr_char_trigram_jaccard = _jaccard(s1_a_trig, cand_a_trig)

        # Street number overlap
        s1_nums = set(extract_numbers(s1_addr))
        cand_nums = set(extract_numbers(cand_addr))
        if s1_nums and cand_nums:
            num_inter = len(s1_nums & cand_nums)
            addr_num_overlap_ratio = num_inter / len(s1_nums)
            addr_any_num_match = 1.0 if num_inter > 0 else 0.0
            # Number count mismatch signal
            addr_num_count_mismatch = 0.0 if len(s1_nums) == len(cand_nums) else 1.0
        else:
            addr_num_overlap_ratio = 0.0
            addr_any_num_match = 0.0
            addr_num_count_mismatch = 0.0

        # Locality token Jaccard (tokens >= 5 chars from both addresses)
        s1_locality = {t for t in extract_address_tokens(s1_addr) if len(t) >= 5}
        cand_locality = {t for t in extract_address_tokens(cand_addr) if len(t) >= 5}
        addr_locality_jaccard = _jaccard(s1_locality, cand_locality)

        # Address length difference
        max_addr_len = max(len(s1_addr), len(cand_addr), 1)
        addr_len_diff = abs(len(s1_addr) - len(cand_addr)) / max_addr_len
    else:
        addr_ratio = 0.0
        addr_token_sort_ratio = 0.0
        addr_token_set_ratio = 0.0
        addr_jaccard_tokens = 0.0
        addr_char_trigram_jaccard = 0.0
        addr_num_overlap_ratio = 0.0
        addr_any_num_match = 0.0
        addr_num_count_mismatch = 0.0
        addr_locality_jaccard = 0.0
        addr_len_diff = 0.0

    # ---------------------------------------------------------------
    # Source indicator
    # ---------------------------------------------------------------
    cand_is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    cand_is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0

    # ---------------------------------------------------------------
    # Combined interaction features
    # ---------------------------------------------------------------
    # Product of best name signal × best address signal
    best_name = max(name_token_set_ratio, name_suffix_stripped_token_set, name_char_trigram_jaccard)
    best_addr = max(addr_token_set_ratio, addr_locality_jaccard, addr_char_trigram_jaccard)
    name_x_addr_score = best_name * best_addr

    # Lightweight blocking score proxy (mimics what blocking index would score)
    blocking_score_proxy = (
        (3.0 * name_clean_exact) +
        (2.0 * name_prefix_match) +
        (2.0 * name_domain_match) +
        (1.5 * addr_any_num_match) +
        (1.0 * (1.0 if name_jaccard_tokens > 0.5 else 0.0))
    ) / 9.5  # normalize to ~[0, 1]

    return [
        name_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_partial_ratio,
        name_jaccard_tokens,
        name_char_trigram_jaccard,
        name_clean_exact,
        name_prefix_match,
        name_domain_match,
        name_len_diff,
        name_bigram_jaccard,
        name_suffix_stripped_token_set,
        addr_is_empty,
        addr_both_empty,
        addr_ratio,
        addr_token_sort_ratio,
        addr_token_set_ratio,
        addr_jaccard_tokens,
        addr_char_trigram_jaccard,
        addr_num_overlap_ratio,
        addr_any_num_match,
        addr_num_count_mismatch,
        addr_locality_jaccard,
        addr_len_diff,
        cand_is_source2,
        cand_is_source3,
        name_x_addr_score,
        blocking_score_proxy,
    ]
