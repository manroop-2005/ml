"""
Unit tests for business name and address preprocessing and normalization.
Validates:
1. Preservation of original raw columns.
2. Creation of separate normalized columns.
3. Case normalization.
4. Smart Unicode normalization (accents stripped for Latin, Indic scripts preserved).
5. Whitespace normalization.
6. Punctuation handling (& -> and, noise symbols stripped).
7. Domain name normalization.
8. Business suffix standardization based on training data.
9. Road abbreviation standardization.
10. Preservation of all address numbers.
11. Preservation of locality, city, and state information.
12. Determinism and idempotency.
"""

import unittest
from src.data_loader import EntityRecord
from src.preprocessing import (
    NormalizedEntityRecord,
    normalize_record,
    normalize_dataframe,
    normalize_business_name,
    normalize_business_address,
    strip_accents_latin,
    normalize_whitespace,
    strip_noise_symbols,
    standardize_business_suffixes,
    standardize_address_abbreviations,
    extract_address_numbers,
    clean_alphanumeric,
)


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing routines."""

    def test_preserves_original_columns_in_record(self):
        rec = EntityRecord(
            entity_id="S1-001",
            business_name="  -- Raw Name Inc.  ",
            business_address="105 ELM ST, MORGANTON, NC",
            country="US"
        )
        norm_rec = normalize_record(rec)

        # Original columns must be completely preserved
        self.assertEqual(norm_rec.entity_id, "S1-001")
        self.assertEqual(norm_rec.business_name, "  -- Raw Name Inc.  ")
        self.assertEqual(norm_rec.business_address, "105 ELM ST, MORGANTON, NC")
        self.assertEqual(norm_rec.country, "US")

        # Separate normalized columns must be populated
        self.assertEqual(norm_rec.normalized_name, "raw name inc")
        self.assertEqual(norm_rec.normalized_address, "105 elm st morganton nc")
        self.assertEqual(norm_rec.clean_name, "rawnameinc")
        self.assertEqual(norm_rec.address_numbers, ("105",))

    def test_preserves_original_columns_in_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed in current environment")

        df = pd.DataFrame([
            {
                "entity_id": "S1-100",
                "business_name": "  Ace Corp  ",
                "business_address": "123 Main Street",
                "country": "US"
            }
        ])
        norm_df = normalize_dataframe(df)

        # Check all original columns present and unchanged
        self.assertEqual(norm_df.iloc[0]["entity_id"], "S1-100")
        self.assertEqual(norm_df.iloc[0]["business_name"], "  Ace Corp  ")
        self.assertEqual(norm_df.iloc[0]["business_address"], "123 Main Street")
        self.assertEqual(norm_df.iloc[0]["country"], "US")

        # Check new columns created
        self.assertIn("normalized_name", norm_df.columns)
        self.assertIn("normalized_address", norm_df.columns)
        self.assertIn("clean_name", norm_df.columns)
        self.assertIn("address_numbers", norm_df.columns)

        self.assertEqual(norm_df.iloc[0]["normalized_name"], "ace corp")
        self.assertEqual(norm_df.iloc[0]["normalized_address"], "123 main st")
        self.assertEqual(norm_df.iloc[0]["clean_name"], "acecorp")
        self.assertEqual(norm_df.iloc[0]["address_numbers"], ("123",))

    def test_case_normalization(self):
        self.assertEqual(normalize_business_name("ACME CORPORATION"), "acme corp")
        self.assertEqual(normalize_business_address("105 ELM STREET"), "105 elm st")

    def test_smart_unicode_normalization_accents(self):
        # Accents in Latin should be stripped to standard letters
        self.assertEqual(normalize_business_name("Group Lio Harbor Ímaging"), "group lio harbor imaging")
        self.assertEqual(normalize_business_name("Clinique Bôw"), "clinique bow")
        self.assertEqual(normalize_business_name("SCI Ptit Àmicale"), "sci ptit amicale")
        self.assertEqual(
            normalize_business_address("175 Boulevard du Président Franklin Roosevelt"),
            "175 blvd du president franklin roosevelt"
        )

    def test_smart_unicode_normalization_preserves_indic_scripts(self):
        # Devanagari matras and conjuncts must NOT be stripped or corrupted
        hindi_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        tamil_name = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"
        self.assertEqual(normalize_business_name(hindi_name), hindi_name)
        self.assertEqual(normalize_business_name(tamil_name), tamil_name)

    def test_whitespace_normalization(self):
        messy_str = "   Acme  \t \n  Ventures   \u00a0 LLC   "
        self.assertEqual(normalize_whitespace(messy_str), "Acme Ventures LLC")
        self.assertEqual(normalize_business_name(messy_str), "acme ventures llc")

    def test_punctuation_handling(self):
        # & -> and
        self.assertEqual(normalize_business_name("Amd & Brothers Limited"), "amd and brothers ltd")
        # @ -> at
        self.assertEqual(normalize_business_name("Coffee @ Morning"), "coffee at morning")
        # Quotes and hyphens
        self.assertEqual(normalize_business_name("“Orelee's” Barbershop"), "orelee s barbershop")

    def test_noise_symbol_stripping(self):
        self.assertEqual(normalize_business_name("-- Holloway Peak Inc Seafood"), "holloway peak inc seafood")
        self.assertEqual(normalize_business_name("<< Team Ecole"), "team ecole")
        self.assertEqual(normalize_business_address("##220 DYER AVENUE"), "220 dyer ave")
        self.assertEqual(normalize_business_name("LLC MILLER CRURECNCYHSARLESH,"), "llc miller crurecncyhsarlesh")

    def test_domain_name_normalization(self):
        self.assertEqual(normalize_business_name("wilfordhancock.com"), "wilfordhancock")
        self.assertEqual(normalize_business_name("www.maurewilliams.in"), "maurewilliams")
        self.assertEqual(normalize_business_name("service-paris.fr"), "service paris")

    def test_business_suffix_standardization(self):
        self.assertEqual(normalize_business_name("Ambika Infotech Private Limited"), "ambika infotech pvt ltd")
        self.assertEqual(normalize_business_name("Sunrise Shree Ventures Pvt. Ltd."), "sunrise shree ventures pvt ltd")
        self.assertEqual(normalize_business_name("Total Chemical Innovations, Incorporated"), "total chemical innovations inc")
        self.assertEqual(normalize_business_name("Styles All Realty Corporation"), "styles all realty corp")
        self.assertEqual(normalize_business_name("Gulf Plains L.L.C."), "gulf plains llc")
        self.assertEqual(normalize_business_name("Classe & Cie S.A.R.L."), "classe and cie sarl")

    def test_address_abbreviations_and_numbers_preservation(self):
        raw_addr = "1795 Westchester Drive, Suite #200, High Point, NC 27262"
        norm_addr = normalize_business_address(raw_addr)
        # Verify numbers 1795, 200, 27262 are preserved in place
        self.assertIn("1795", norm_addr)
        self.assertIn("200", norm_addr)
        self.assertIn("27262", norm_addr)
        # Verify abbreviations
        self.assertIn("dr", norm_addr)
        self.assertIn("ste", norm_addr)
        # Verify locality
        self.assertIn("high point nc", norm_addr)

        # Check extracted numbers list
        nums = extract_address_numbers(raw_addr)
        self.assertEqual(nums, ["1795", "200", "27262"])

    def test_complex_multilingual_indian_address(self):
        raw_addr = "AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश"
        norm_addr = normalize_business_address(raw_addr)
        self.assertIn("0684", norm_addr)
        self.assertIn("989", norm_addr)
        self.assertIn("9487203", norm_addr)
        self.assertIn("ghaziabad", norm_addr)
        self.assertIn("उत्तर प्रदेश", norm_addr)

    def test_empty_inputs(self):
        self.assertEqual(normalize_business_name(""), "")
        self.assertEqual(normalize_business_name(None), "")
        self.assertEqual(normalize_business_address(""), "")
        self.assertEqual(normalize_business_address(None), "")
        self.assertEqual(extract_address_numbers(""), [])

    def test_determinism_and_idempotency(self):
        name = "  -- Total Chemical Innovations, Incorporated & Co.  "
        addr = "1795 Westchester Drive, Suite #200, High Point, NC 27262"

        norm_name_1 = normalize_business_name(name)
        norm_name_2 = normalize_business_name(norm_name_1)
        self.assertEqual(norm_name_1, norm_name_2)

        norm_addr_1 = normalize_business_address(addr)
        norm_addr_2 = normalize_business_address(norm_addr_1)
        self.assertEqual(norm_addr_1, norm_addr_2)


if __name__ == "__main__":
    unittest.main()
