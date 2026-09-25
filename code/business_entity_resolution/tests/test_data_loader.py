"""
Unit tests for data_loader module.
Uses standard library unittest so it runs everywhere with or without pytest.

Tests:
1. Loading actual challenge TSV files (train/test).
2. Preserving raw values and handling empty addresses.
3. Catching missing columns and comma-separated file errors.
4. Validating S1/S2/S3 prefix enforcement.
5. Detecting malformed rows and useful error messages.
6. Validating ground truth parsing, singletons, and integrity checks.
7. Verifying no silent dropping of rows.
8. Configurable paths and limits.
"""

import os
import tempfile
import unittest
from pathlib import Path

from src.data_loader import (
    EntityRecord,
    GroundTruthRecord,
    stream_source_file,
    load_source_file,
    stream_ground_truth,
    load_ground_truth,
    load_dataset_split,
    DataLoaderError,
    FileFormatError,
    MissingColumnError,
    MalformedRowError,
    InvalidEntityIdError,
    GroundTruthFormatError,
    SOURCE_REQUIRED_COLUMNS,
    GROUND_TRUTH_REQUIRED_COLUMNS,
)

# Reference dataset path relative to repository
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAIN_DIR = REPO_ROOT / "dataset" / "train"
TEST_DIR = REPO_ROOT / "dataset" / "test"


class TestSourceDataLoader(unittest.TestCase):
    """Tests for loading source files."""

    def test_load_real_train_source1(self):
        s1_file = TRAIN_DIR / "train_source1.tsv"
        records = load_source_file(s1_file, expected_prefix="S1-", limit=10)
        self.assertEqual(len(records), 10)
        for r in records:
            self.assertIsInstance(r, EntityRecord)
            self.assertTrue(r.entity_id.startswith("S1-"))
            self.assertGreater(len(r.business_name), 0)
            self.assertGreater(len(r.business_address), 0)
            self.assertIn(r.country, ("US", "India"))

    def test_load_real_test_source1_with_france(self):
        s1_file = TEST_DIR / "test_source1.tsv"
        records = load_source_file(s1_file, expected_prefix="S1-", limit=100)
        self.assertEqual(len(records), 100)
        countries = {r.country for r in records}
        self.assertTrue("US" in countries or "India" in countries or "France" in countries)

    def test_load_real_train_source2_and_source3(self):
        s2_file = TRAIN_DIR / "train_source2.tsv"
        s3_file = TRAIN_DIR / "train_source3.tsv"

        s2_records = load_source_file(s2_file, expected_prefix="S2-", limit=10)
        self.assertEqual(len(s2_records), 10)
        self.assertTrue(all(r.entity_id.startswith("S2-") for r in s2_records))

        s3_records = load_source_file(s3_file, expected_prefix="S3-", limit=10)
        self.assertEqual(len(s3_records), 10)
        self.assertTrue(all(r.entity_id.startswith("S3-") for r in s3_records))

    def test_preserves_raw_values_and_empty_address(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            tf.write("S1-001\t  Leading and Trailing Spaces Inc  \t\tUS\n")
            tf.write("S1-002\tSpecial & Co / Pvt Ltd\tSuite #102, 1st Fl.\tIndia\n")
            tf.flush()
            temp_path = tf.name

        try:
            records = load_source_file(temp_path, expected_prefix="S1-")
            self.assertEqual(len(records), 2)
            # Check raw whitespace preserved
            self.assertEqual(records[0].business_name, "  Leading and Trailing Spaces Inc  ")
            self.assertEqual(records[0].business_address, "")
            self.assertEqual(records[0].country, "US")
            # Check symbols preserved
            self.assertEqual(records[1].business_name, "Special & Co / Pvt Ltd")
            self.assertEqual(records[1].business_address, "Suite #102, 1st Fl.")
        finally:
            os.remove(temp_path)

    def test_as_dataframe_no_nan_coercion(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed in the active Python environment")

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            tf.write("S2-001\tNA\t\tUS\n")
            tf.flush()
            temp_path = tf.name

        try:
            df = load_source_file(temp_path, as_dataframe=True)
            self.assertEqual(len(df), 1)
            # Must remain string 'NA', not NaN or float
            self.assertEqual(df.iloc[0]["business_name"], "NA")
            self.assertEqual(df.iloc[0]["business_address"], "")
        finally:
            os.remove(temp_path)

    def test_detects_comma_separated_file(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("entity_id,business_name,business_address,country\n")
            tf.write("S1-001,Acme Corp,123 Main St,US\n")
            tf.flush()
            temp_path = tf.name

        try:
            with self.assertRaises(FileFormatError) as ctx:
                load_source_file(temp_path)
            self.assertIn("contains commas but no TAB", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_detects_missing_columns(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            # Missing 'country' column
            tf.write("entity_id\tbusiness_name\tbusiness_address\n")
            tf.write("S1-001\tAcme Corp\t123 Main St\n")
            tf.flush()
            temp_path = tf.name

        try:
            with self.assertRaises(MissingColumnError) as ctx:
                load_source_file(temp_path)
            self.assertIn("invalid header columns", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_detects_malformed_row_column_count(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            tf.write("S1-001\tAcme Corp\t123 Main St\tUS\n")
            # Line 3 is missing a column
            tf.write("S1-002\tBad Row\tUS\n")
            tf.flush()
            temp_path = tf.name

        try:
            with self.assertRaises(MalformedRowError) as ctx:
                load_source_file(temp_path)
            err = ctx.exception
            self.assertEqual(err.line_number, 3)
            self.assertIn("Expected 4 columns, found 3", str(err))
        finally:
            os.remove(temp_path)

    def test_detects_invalid_entity_id_prefix(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            # Wrong prefix: S2- in a file expecting S1-
            tf.write("S2-99999\tAcme Corp\t123 Main St\tUS\n")
            tf.flush()
            temp_path = tf.name

        try:
            with self.assertRaises(InvalidEntityIdError) as ctx:
                load_source_file(temp_path, expected_prefix="S1-")
            err = ctx.exception
            self.assertEqual(err.line_number, 2)
            self.assertIn("S2-99999", str(err))
            self.assertIn("Expected prefix: 'S1-'", str(err))
        finally:
            os.remove(temp_path)


class TestGroundTruthDataLoader(unittest.TestCase):
    """Tests for loading ground truth files."""

    def test_load_real_train_ground_truth(self):
        gt_file = TRAIN_DIR / "train_ground_truth.tsv"
        records = load_ground_truth(gt_file, limit=20, as_dict=False)
        self.assertEqual(len(records), 20)
        for r in records:
            self.assertIsInstance(r, GroundTruthRecord)
            self.assertTrue(r.source1_entity_id.startswith("S1-"))
            for mid in r.matched_entity_ids:
                self.assertTrue(mid.startswith(("S2-", "S3-")))

    def test_load_ground_truth_as_dict(self):
        gt_file = TRAIN_DIR / "train_ground_truth.tsv"
        mapping = load_ground_truth(gt_file, limit=20, as_dict=True)
        self.assertIsInstance(mapping, dict)
        self.assertEqual(len(mapping), 20)
        for s1, matches in mapping.items():
            self.assertTrue(s1.startswith("S1-"))
            self.assertIsInstance(matches, set)

    def test_handles_singletons(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("source1_entity_id\tmatched_entity_ids\n")
            tf.write("S1-001\tS2-001,S3-002\n")
            tf.write("S1-002\t\n")  # Singleton
            tf.flush()
            temp_path = tf.name

        try:
            mapping = load_ground_truth(temp_path, as_dict=True)
            self.assertEqual(mapping["S1-001"], {"S2-001", "S3-002"})
            self.assertEqual(mapping["S1-002"], set())  # Empty set for singleton
        finally:
            os.remove(temp_path)

    def test_detects_self_match_in_ground_truth(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("source1_entity_id\tmatched_entity_ids\n")
            # S1-001 self-matches S1-002
            tf.write("S1-001\tS1-002,S2-003\n")
            tf.flush()
            temp_path = tf.name

        try:
            with self.assertRaises(GroundTruthFormatError) as ctx:
                load_ground_truth(temp_path)
            self.assertIn("Self-match error", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_detects_intra_list_duplicates(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("source1_entity_id\tmatched_entity_ids\n")
            # S2-001 appears twice
            tf.write("S1-001\tS2-001,S2-001\n")
            tf.flush()
            temp_path = tf.name

        try:
            with self.assertRaises(GroundTruthFormatError) as ctx:
                load_ground_truth(temp_path)
            self.assertIn("Repeated ID within match list", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_detects_duplicate_s1_rows(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tf:
            tf.write("source1_entity_id\tmatched_entity_ids\n")
            tf.write("S1-001\tS2-001\n")
            tf.write("S1-001\tS3-002\n")  # Duplicate S1 row
            tf.flush()
            temp_path = tf.name

        try:
            with self.assertRaises(GroundTruthFormatError) as ctx:
                load_ground_truth(temp_path)
            self.assertIn("Duplicate source1_entity_id", str(ctx.exception))
        finally:
            os.remove(temp_path)


class TestConfigurablePaths(unittest.TestCase):
    """Tests for path configurability and whole-split loading."""

    def test_load_dataset_split(self):
        data = load_dataset_split(split_dir=TRAIN_DIR, is_train=True, limit_per_file=5)
        self.assertIn("source1", data)
        self.assertIn("source2", data)
        self.assertIn("source3", data)
        self.assertIn("ground_truth", data)

        self.assertEqual(len(data["source1"]), 5)
        self.assertEqual(len(data["source2"]), 5)
        self.assertEqual(len(data["source3"]), 5)
        self.assertEqual(len(data["ground_truth"]), 5)


if __name__ == "__main__":
    unittest.main()
