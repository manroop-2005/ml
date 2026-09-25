"""
Inference pipeline for Business Entity Resolution.
Generates candidates, predicts match probabilities using the trained GBDT model,
and outputs matching_results.tsv and candidate_pairs.tsv adhering to the exact challenge format.
"""

import os
import gc
import csv
import sys
import time
import argparse
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from .config import (
    TEST_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
    MAX_CANDIDATES_PER_ENTITY
)
from .blocking import BlockingIndex
from .features import compute_pair_features
from .model import EntityResolutionModel


def get_all_s1_records(
    test_dir: str,
    country_filter: Optional[str] = None,
    limit: int = 0
) -> List[Tuple[str, str, str, str]]:
    """Read Source 1 test records."""
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    records = []
    print(f"Reading Source 1 test records from {s1_path}...")
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row or len(row) < 4:
                continue
            if country_filter and row[3] != country_filter:
                continue
            records.append((row[0], row[1], row[2], row[3]))
            if limit > 0 and len(records) >= limit:
                break
    print(f"Loaded {len(records)} test S1 records.")
    return records


def build_country_target_index(
    test_dir: str,
    target_country: str,
    max_candidates: int = MAX_CANDIDATES_PER_ENTITY,
    target_limit: int = 0
) -> BlockingIndex:
    """Build blocking index for Source 2 and Source 3 records belonging to target_country."""
    index = BlockingIndex(max_candidates=max_candidates)
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for path in [s2_path, s3_path]:
        if not os.path.exists(path):
            continue
        print(f"Indexing target records from {os.path.basename(path)} for country '{target_country}'...")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                if row[3] == target_country:
                    index.index_target_record(row[0], row[1], row[2], row[3])
                    count += 1
                    if target_limit > 0 and count >= target_limit:
                        break
        print(f"  Loaded {count} target records from {os.path.basename(path)}.")

    index.prune_frequent_keys()
    print(f"Total target records indexed for country '{target_country}': {len(index.records)}")
    return index


def predict_pipeline(
    test_dir: str = TEST_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    country_filter: Optional[str] = None,
    quick_limit: int = 0,
    batch_size: int = 2000
):
    """Run full candidate generation and model scoring for test set."""
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 60)
    print("Starting ML Challenge 2026 Entity Resolution Inference")
    print(f"Test directory:   {test_dir}")
    print(f"Model artifact:   {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Quick limit:      {quick_limit} (0 = all)")
    print("=" * 60)

    # 1. Load trained model
    print(f"Loading model artifact from {model_path}...")
    model = EntityResolutionModel.load(model_path)
    print(f"Loaded model successfully. Decision threshold: {model.threshold:.3f}")

    # 2. Read S1 test records
    all_s1_records = get_all_s1_records(test_dir, country_filter=country_filter, limit=quick_limit)

    # Group S1 by country
    s1_by_country: Dict[str, List[Tuple[str, str, str]]] = {}
    for s1_id, name, addr, country in all_s1_records:
        if country not in s1_by_country:
            s1_by_country[country] = []
        s1_by_country[country].append((s1_id, name, addr))

    print(f"Countries found in test S1: {sorted(s1_by_country.keys())}")
    print(f"NOTE: All countries (including France) are handled dynamically.")

    # Results dictionary: s1_id -> (candidate_ids_list, matched_ids_list)
    results: Dict[str, Tuple[List[str], List[str]]] = {}

    target_limit = quick_limit * 20 if quick_limit > 0 else 0

    # 3. Process each country independently to preserve memory
    for country, s1_list in s1_by_country.items():
        print(f"\n--- Processing Country: {country} ({len(s1_list)} S1 entities) ---")
        t_country_start = time.time()

        # Build target index for this country
        index = build_country_target_index(test_dir, target_country=country, target_limit=target_limit)

        # Batch process S1 records
        total_s1 = len(s1_list)
        for start_idx in range(0, total_s1, batch_size):
            end_idx = min(start_idx + batch_size, total_s1)
            batch = s1_list[start_idx:end_idx]

            batch_pairs = []
            batch_cand_ids = []
            batch_s1_ids = []

            for s1_id, name, addr in batch:
                candidates = index.query_candidates(name, addr)
                if not candidates:
                    results[s1_id] = ([], [])
                    continue

                for cid in candidates:
                    if cid in index.records:
                        cand_name, cand_addr, _ = index.records[cid]
                        feats = compute_pair_features(name, addr, cid, cand_name, cand_addr)
                        batch_pairs.append(feats)
                        batch_cand_ids.append(cid)
                        batch_s1_ids.append(s1_id)

                results[s1_id] = (candidates, [])

            if batch_pairs:
                X_batch = np.array(batch_pairs, dtype=np.float32)
                probs = model.predict_proba(X_batch)

                # Group predictions by s1_id
                for s1_id, cid, prob in zip(batch_s1_ids, batch_cand_ids, probs):
                    if prob >= model.threshold:
                        cands, matches = results[s1_id]
                        matches.append(cid)

            if end_idx % 10000 == 0 or end_idx == total_s1:
                elapsed = time.time() - t_country_start
                rate = end_idx / max(elapsed, 0.001)
                print(f"  Processed {end_idx}/{total_s1} entities ({rate:.0f} entities/sec)")

        # Clean up country index from memory
        del index
        gc.collect()

    # 4. Write matching_results.tsv and candidate_pairs.tsv
    print(f"\nWriting final results to {matching_out_path} and {candidate_out_path}...")

    # Write in the original order of all_s1_records
    total_matches = 0
    total_candidates = 0
    singletons = 0

    with open(matching_out_path, "w", encoding="utf-8", newline="") as f_match, \
         open(candidate_out_path, "w", encoding="utf-8", newline="") as f_cand:

        # Headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id, _, _, _ in all_s1_records:
            cand_list, match_list = results.get(s1_id, ([], []))

            # Ensure matches are subset of candidates and deduplicated
            clean_cands = []
            seen_c = set()
            for c in cand_list:
                if c not in seen_c:
                    seen_c.add(c)
                    clean_cands.append(c)

            clean_matches = []
            seen_m = set()
            for m in match_list:
                if m not in seen_m and m in seen_c:
                    seen_m.add(m)
                    clean_matches.append(m)

            if not clean_matches:
                singletons += 1

            total_matches += len(clean_matches)
            total_candidates += len(clean_cands)

            cand_str = ",".join(clean_cands)
            match_str = ",".join(clean_matches)

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    total_s1_count = len(all_s1_records)
    dt = time.time() - t0
    print("-" * 60)
    print("Inference Summary:")
    print(f"  Total Source 1 entities processed: {total_s1_count}")
    print(f"  Total candidates generated:        {total_candidates} (avg {total_candidates/max(total_s1_count, 1):.1f}/entity)")
    print(f"  Total matches predicted:           {total_matches} (avg {total_matches/max(total_s1_count, 1):.1f}/entity)")
    print(f"  Predicted singletons (no matches): {singletons} ({singletons/max(total_s1_count, 1):.2%})")
    print(f"  Total pipeline runtime:            {dt:.1f} seconds")
    print("-" * 60)
    return matching_out_path, candidate_out_path


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for Entity Resolution")
    parser.add_argument("--test-dir", default=TEST_DIR, help="Path to test dataset directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to trained model artifact")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output folder for results")
    parser.add_argument("--country", default=None, help="Filter to specific country")
    parser.add_argument("--quick-limit", type=int, default=0, help="Limit number of test S1 records (0 = all)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for feature scoring")
    args = parser.parse_args()

    predict_pipeline(
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        country_filter=args.country,
        quick_limit=args.quick_limit,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
