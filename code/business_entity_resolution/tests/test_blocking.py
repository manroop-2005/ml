"""
Unit tests for multi-strategy candidate generation / blocking.
Tests:
1. Extraction of blocking keys across multiple strategies.
2. Inverted index construction and frequency pruning.
3. Combining candidates across strategies without duplicate candidate pairs.
4. Strategy provenance tracking (recording which strategy generated each candidate).
5. Candidate set size constraints (max_candidates).
6. Support for both S1 -> S2 and S1 -> S3 candidates.
7. Handling missing addresses and noisy inputs gracefully.
"""

import unittest
from src.blocking import (
    MultiStrategyBlockingIndex,
    extract_blocking_keys,
    STRATEGY_EXACT_NAME,
    STRATEGY_NAME_PREFIX,
    STRATEGY_NAME_TOKENS,
    STRATEGY_ADDR_COMPOUND,
    STRATEGY_DOMAIN_CORE,
    STRATEGY_ADDR_NUMBERS,
    STRATEGY_ADDR_LOCALITY,
    ALL_STRATEGIES,
)


class TestBlockingKeyExtraction(unittest.TestCase):
    """Test individual blocking strategy key extraction."""

    def test_extract_all_strategy_keys(self):
        name = "Orelee's Barbershop Inc"
        addr = "1795 Westchester Drive, High Point, NC"
        keys = extract_blocking_keys(name, addr)

        # Check exact name & prefix
        self.assertIn(STRATEGY_EXACT_NAME, keys)
        self.assertIn("oreleesbarbershopinc", keys[STRATEGY_EXACT_NAME])
        self.assertIn(STRATEGY_NAME_PREFIX, keys)
        self.assertIn("orelee", keys[STRATEGY_NAME_PREFIX])

        # Check name tokens (excluding 'inc')
        self.assertIn(STRATEGY_NAME_TOKENS, keys)
        tokens = keys[STRATEGY_NAME_TOKENS]
        self.assertTrue("orelee" in tokens or "orelees" in tokens)
        self.assertIn("barbershop", tokens)
        self.assertNotIn("inc", tokens)

        # Check address compound and numbers
        self.assertIn(STRATEGY_ADDR_COMPOUND, keys)
        self.assertIn("1795_westchester", keys[STRATEGY_ADDR_COMPOUND])
        self.assertIn(STRATEGY_ADDR_NUMBERS, keys)
        self.assertIn("1795", keys[STRATEGY_ADDR_NUMBERS])

    def test_extract_domain_core_strategy(self):
        name = "wilfordhancock.com"
        addr = "Mack Rd, Haltom City, Texas"
        keys = extract_blocking_keys(name, addr)

        self.assertIn(STRATEGY_DOMAIN_CORE, keys)
        self.assertIn("wilfordhancock", keys[STRATEGY_DOMAIN_CORE])

    def test_empty_address_handling(self):
        name = "Acme Ventures LLC"
        addr = ""  # Missing address
        keys = extract_blocking_keys(name, addr)

        # Name strategies must still work
        self.assertIn(STRATEGY_EXACT_NAME, keys)
        self.assertIn(STRATEGY_NAME_PREFIX, keys)
        self.assertIn(STRATEGY_NAME_TOKENS, keys)
        # Address strategies should be empty
        self.assertEqual(len(keys[STRATEGY_ADDR_COMPOUND]), 0)
        self.assertEqual(len(keys[STRATEGY_ADDR_NUMBERS]), 0)


class TestMultiStrategyBlockingIndex(unittest.TestCase):
    """Test multi-strategy index construction, querying, and provenance."""

    def setUp(self):
        self.idx = MultiStrategyBlockingIndex(max_candidates=10)
        # Target records from S2 and S3
        self.idx.index_target_record("S2-001", "Orelee Barbershop", "1795 Westchester Dr", "US")
        self.idx.index_target_record("S3-002", "Orelee Hair Salon", "1795 Westchester Dr", "US")
        self.idx.index_target_record("S2-003", "wilfordhancock.com", "Mack Rd, Haltom City", "US")
        self.idx.index_target_record("S3-004", "Dréxkor Barber", "1795 Westchester Drive", "US")
        self.idx.index_target_record("S2-005", "Totally Unrelated", "999 Other St", "US")

    def test_query_retrieves_s2_and_s3_without_duplicates(self):
        query_name = "Orelee's Barbershop Inc"
        query_addr = "1795 Westchester Drive, High Point, NC"

        cands = self.idx.query_candidates(query_name, query_addr)
        # Check no duplicates in candidate list
        self.assertEqual(len(cands), len(set(cands)))
        # Check both S2 and S3 candidates retrieved
        s2_cands = [c for c in cands if c.startswith("S2-")]
        s3_cands = [c for c in cands if c.startswith("S3-")]
        self.assertGreater(len(s2_cands), 0)
        self.assertGreater(len(s3_cands), 0)

        # Expected true matches should be present
        self.assertIn("S2-001", cands)
        self.assertIn("S3-002", cands)

    def test_provenance_tracking(self):
        query_name = "Orelee's Barbershop Inc"
        query_addr = "1795 Westchester Drive, High Point, NC"

        provenance = self.idx.query_candidates_with_provenance(query_name, query_addr)
        self.assertIn("S2-001", provenance)

        # S2-001 matches on name tokens AND address compound
        s2_strats = provenance["S2-001"]
        self.assertIsInstance(s2_strats, set)
        self.assertTrue(
            STRATEGY_NAME_TOKENS in s2_strats or
            STRATEGY_ADDR_COMPOUND in s2_strats or
            STRATEGY_ADDR_NUMBERS in s2_strats
        )

        # S3-004 has completely different fantasy name 'Dréxkor Barber', but identical address
        self.assertIn("S3-004", provenance)
        s3_strats = provenance["S3-004"]
        self.assertTrue(
            STRATEGY_ADDR_COMPOUND in s3_strats or
            STRATEGY_ADDR_NUMBERS in s3_strats
        )

    def test_domain_matching_strategy(self):
        query_name = "Wilford Hancock"
        query_addr = "Mack Road, Haltom City, Texas"

        provenance = self.idx.query_candidates_with_provenance(query_name, query_addr)
        self.assertIn("S2-003", provenance)
        self.assertIn(STRATEGY_DOMAIN_CORE, provenance["S2-003"])

    def test_max_candidates_cap(self):
        small_idx = MultiStrategyBlockingIndex(max_candidates=2)
        small_idx.index_target_record("S2-001", "Acme Alpha", "123 Main St", "US")
        small_idx.index_target_record("S2-002", "Acme Beta", "123 Main St", "US")
        small_idx.index_target_record("S3-003", "Acme Gamma", "123 Main St", "US")

        cands = small_idx.query_candidates("Acme Corporation", "123 Main St")
        self.assertEqual(len(cands), 2)  # Strictly capped at 2

    def test_prune_frequent_keys(self):
        idx = MultiStrategyBlockingIndex()
        # Add 10 records with the exact same name token 'market'
        for i in range(10):
            idx.index_target_record(f"S2-{i:03d}", "Super Market", f"{i} Street", "US")

        # Prune keys with > 5 postings
        pruned = idx.prune_frequent_keys(max_postings=5)
        self.assertGreater(pruned, 0)
        # Frequent key should be gone from the index
        self.assertNotIn("market", idx.index[STRATEGY_NAME_TOKENS])


if __name__ == "__main__":
    unittest.main()
