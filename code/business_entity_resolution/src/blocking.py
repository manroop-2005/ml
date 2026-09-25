"""
Blocking / Candidate Generation module for Business Entity Resolution.

Requirements addressed:
1. Uses only provided challenge data without external APIs/databases.
2. Implements multiple complementary blocking strategies:
   - exact_clean_name: exact match on normalized/cleaned business name
   - name_prefix_6: 6-character prefix of clean alphanumeric name
   - name_distinctive_tokens: informative word tokens (excluding legal suffixes)
   - address_number_and_street: compound key combining street number and street token
   - domain_name_core: website / domain name core alignment
   - address_numbers: distinctive numeric building/PIN identifiers
3. Prioritizes high recall so true matches are preserved.
4. Keeps candidate sets small via frequency-based key pruning and preliminary scoring.
5. Combines candidates across all strategies and removes duplicate candidate pairs.
6. Tracks which blocking strategy generated each candidate (provenance tracking).
7. Supports both S1 -> S2 and S1 -> S3 matching.
8. Test-time generation operates strictly without ground-truth labels.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

try:
    from .config import MAX_CANDIDATES_PER_ENTITY, MAX_BLOCK_POSTINGS, MIN_TOKEN_LEN
    from .preprocessing import (
        clean_alphanumeric,
        extract_address_tokens,
        extract_address_numbers,
        extract_domain_core,
        extract_name_tokens,
        normalize_business_name,
        normalize_business_address,
    )
    from .data_loader import EntityRecord
except ImportError:
    from config import MAX_CANDIDATES_PER_ENTITY, MAX_BLOCK_POSTINGS, MIN_TOKEN_LEN
    from preprocessing import (
        clean_alphanumeric,
        extract_address_tokens,
        extract_address_numbers,
        extract_domain_core,
        extract_name_tokens,
        normalize_business_name,
        normalize_business_address,
    )
    from data_loader import EntityRecord


# Strategy Identifiers
STRATEGY_EXACT_NAME = "exact_clean_name"
STRATEGY_NAME_PREFIX = "name_prefix_6"
STRATEGY_NAME_TOKENS = "name_distinctive_tokens"
STRATEGY_ADDR_COMPOUND = "address_number_and_street"
STRATEGY_DOMAIN_CORE = "domain_name_core"
STRATEGY_ADDR_NUMBERS = "address_numbers"
# New: pure address locality tokens (no number required) — catches transliteration misses
STRATEGY_ADDR_LOCALITY = "address_locality_tokens"

ALL_STRATEGIES: Tuple[str, ...] = (
    STRATEGY_EXACT_NAME,
    STRATEGY_NAME_PREFIX,
    STRATEGY_NAME_TOKENS,
    STRATEGY_ADDR_COMPOUND,
    STRATEGY_DOMAIN_CORE,
    STRATEGY_ADDR_NUMBERS,
    STRATEGY_ADDR_LOCALITY,
)

# Relative strategy weights for preliminary candidate ranking
STRATEGY_WEIGHTS: Dict[str, int] = {
    STRATEGY_EXACT_NAME: 6,
    STRATEGY_DOMAIN_CORE: 5,
    STRATEGY_ADDR_COMPOUND: 4,
    STRATEGY_NAME_TOKENS: 3,
    STRATEGY_NAME_PREFIX: 2,
    STRATEGY_ADDR_NUMBERS: 2,
    # Lower weight — locality tokens alone are less discriminative
    STRATEGY_ADDR_LOCALITY: 2,
}


def extract_blocking_keys(name: str, addr: str) -> Dict[str, Set[str]]:
    """
    Extract blocking keys for each strategy from a name and address.
    Returns: {strategy_name: set_of_keys}
    """
    strategy_keys: Dict[str, Set[str]] = defaultdict(set)

    # 1. Clean alphanumeric name & prefix (both raw and suffix-stripped)
    raw_clean = clean_alphanumeric(name)
    norm_name = normalize_business_name(name)
    norm_clean = clean_alphanumeric(norm_name)

    for n_clean in {raw_clean, norm_clean}:
        if len(n_clean) >= MIN_TOKEN_LEN:
            strategy_keys[STRATEGY_EXACT_NAME].add(n_clean[:25])
            if len(n_clean) >= 5:
                strategy_keys[STRATEGY_NAME_PREFIX].add(n_clean[:6])

    # 2. Distinctive name tokens
    name_tokens = extract_name_tokens(name)
    for tok in name_tokens:
        if len(tok) >= MIN_TOKEN_LEN:
            strategy_keys[STRATEGY_NAME_TOKENS].add(tok)

    # 3. Domain core if name represents a website
    dom = extract_domain_core(name)
    if dom and len(dom) >= 4:
        strategy_keys[STRATEGY_DOMAIN_CORE].add(dom)

    # 4. Address numbers & compound keys
    if addr:
        nums = extract_address_numbers(addr)
        addr_toks = extract_address_tokens(addr)

        if nums and addr_toks:
            for num in nums[:2]:
                for atok in addr_toks[:2]:
                    strategy_keys[STRATEGY_ADDR_COMPOUND].add(f"{num}_{atok}")

        for num in nums[:2]:
            if len(num) >= 3:
                strategy_keys[STRATEGY_ADDR_NUMBERS].add(num)

        # 5. Pure locality tokens — distinctive address words without needing a number.
        #    Use longer tokens (>= 5 chars) to remain discriminative.
        #    This catches transliterated name pairs that share city/locality names.
        for tok in addr_toks:
            if len(tok) >= 5:
                strategy_keys[STRATEGY_ADDR_LOCALITY].add(tok)

    return strategy_keys


class MultiStrategyBlockingIndex:
    """
    Inverted index partitioned by blocking strategy.
    Maps (strategy, key) -> list of target entity IDs (Source 2 and Source 3).
    Tracks which strategies generated which candidates (provenance).
    """

    def __init__(self, max_candidates: int = MAX_CANDIDATES_PER_ENTITY):
        self.max_candidates = max_candidates
        # strategy -> {key -> list of target entity IDs}
        self.index: Dict[str, Dict[str, List[str]]] = {
            strat: defaultdict(list) for strat in ALL_STRATEGIES
        }
        # Cache target records: entity_id -> (name, addr, country)
        self.records: Dict[str, Tuple[str, str, str]] = {}

    def index_target_record(self, entity_id: str, name: str, addr: str, country: str):
        """Add a target record (Source 2 or Source 3) to the multi-strategy index."""
        self.records[entity_id] = (name, addr, country)
        strategy_keys = extract_blocking_keys(name, addr)

        for strat, keys in strategy_keys.items():
            for key in keys:
                self.index[strat][key].append(entity_id)

    def prune_frequent_keys(self, max_postings: int = MAX_BLOCK_POSTINGS):
        """
        Prune uninformative high-frequency keys from all strategy indices
        to prevent combinatorial explosion and keep candidate sets tight.
        """
        pruned_count = 0
        for strat in ALL_STRATEGIES:
            keys_to_del = [k for k, postings in self.index[strat].items() if len(postings) > max_postings]
            for k in keys_to_del:
                del self.index[strat][k]
                pruned_count += 1
        return pruned_count

    def query_candidates_with_provenance(
        self,
        name: str,
        addr: str,
        max_candidates: Optional[int] = None
    ) -> Dict[str, Set[str]]:
        """
        Retrieve candidate target IDs for a query record (Source 1),
        along with the set of blocking strategies that retrieved each candidate.

        Returns:
            {candidate_id: set_of_strategies_that_fired}
        """
        limit = max_candidates if max_candidates is not None else self.max_candidates
        query_keys = extract_blocking_keys(name, addr)

        # candidate_id -> set of strategy names
        candidate_provenance: Dict[str, Set[str]] = defaultdict(set)
        # candidate_id -> accumulated score for ranking
        candidate_scores: Dict[str, int] = defaultdict(int)

        for strat, keys in query_keys.items():
            strat_index = self.index.get(strat, {})
            weight = STRATEGY_WEIGHTS.get(strat, 1)

            for key in keys:
                postings = strat_index.get(key)
                if postings:
                    for cid in postings:
                        candidate_provenance[cid].add(strat)
                        candidate_scores[cid] += weight

        # Bidirectional Domain Core Cross-Matching:
        # 1. Query name -> check if clean name matches any target's domain core
        dom_index = self.index.get(STRATEGY_DOMAIN_CORE, {})
        if dom_index:
            raw_clean = clean_alphanumeric(name)
            norm_clean = clean_alphanumeric(normalize_business_name(name))
            for probe in {raw_clean, norm_clean}:
                if probe and len(probe) >= 5:
                    domain_postings = dom_index.get(probe[:25])
                    if domain_postings:
                        for cid in domain_postings:
                            candidate_provenance[cid].add(STRATEGY_DOMAIN_CORE)
                            candidate_scores[cid] += STRATEGY_WEIGHTS.get(STRATEGY_DOMAIN_CORE, 5)

        # 2. Query domain -> check if query domain core matches any target's clean name
        query_dom = extract_domain_core(name)
        if query_dom and len(query_dom) >= 4:
            exact_index = self.index.get(STRATEGY_EXACT_NAME, {})
            if exact_index:
                exact_postings = exact_index.get(query_dom[:25])
                if exact_postings:
                    for cid in exact_postings:
                        candidate_provenance[cid].add(STRATEGY_DOMAIN_CORE)
                        candidate_scores[cid] += STRATEGY_WEIGHTS.get(STRATEGY_DOMAIN_CORE, 5)

        if not candidate_provenance:
            return {}

        # If a limit is set, rank candidates by score descending
        if limit > 0 and len(candidate_provenance) > limit:
            sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            return {cid: candidate_provenance[cid] for cid, _ in sorted_cands}

        return dict(candidate_provenance)

    def query_candidates(
        self,
        name: str,
        addr: str,
        max_candidates: Optional[int] = None
    ) -> List[str]:
        """
        Retrieve a deduplicated list of candidate target IDs for a query record.
        Ranked by multi-strategy match strength.
        """
        provenance = self.query_candidates_with_provenance(name, addr, max_candidates=max_candidates)
        return list(provenance.keys())


# Alias for backwards compatibility with earlier model and predict imports
BlockingIndex = MultiStrategyBlockingIndex
