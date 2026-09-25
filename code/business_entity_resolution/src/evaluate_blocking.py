"""
Evaluation script to measure blocking recall and candidate reduction on training/validation data.

Metrics measured:
1. True Match Recall (percentage of ground truth matches retained by candidates).
2. S1 -> S2 Recall vs S1 -> S3 Recall.
3. Average, Median, and Maximum candidate count per S1 entity.
4. Candidate Reduction Ratio compared with exhaustive Cartesian comparison.
5. Individual Recall by Blocking Strategy (Exact Name, Prefix, Tokens, Address Compound, Domain).
6. Multi-strategy overlap and provenance analysis.
"""

import os
import csv
import sys
import time
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

try:
    from .config import DATA_DIR, TRAIN_DIR, MAX_CANDIDATES_PER_ENTITY
    from .blocking import (
        MultiStrategyBlockingIndex,
        ALL_STRATEGIES,
        STRATEGY_WEIGHTS,
        extract_blocking_keys,
    )
    from .data_loader import load_source_file, load_ground_truth
except ImportError:
    from config import DATA_DIR, TRAIN_DIR, MAX_CANDIDATES_PER_ENTITY
    from blocking import (
        MultiStrategyBlockingIndex,
        ALL_STRATEGIES,
        STRATEGY_WEIGHTS,
        extract_blocking_keys,
    )
    from data_loader import load_source_file, load_ground_truth


def evaluate_blocking(
    train_dir: str = TRAIN_DIR,
    sample_size: int = 3000,
    max_candidates: int = MAX_CANDIDATES_PER_ENTITY,
    background_target_sample: int = 100000,
    seed: int = 42
) -> Dict:
    """
    Run an empirical blocking evaluation on a sample of training entities and ground truth.
    """
    print("=" * 70)
    print("Evaluating Candidate Generation (Blocking) Strategies")
    print(f"Data directory:           {train_dir}")
    print(f"Evaluation sample size:   {sample_size} S1 entities")
    print(f"Max candidates per S1:    {max_candidates}")
    print("=" * 70)

    t0 = time.time()

    # 1. Load S1 sample
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"Loading {sample_size} Source 1 entities...")
    s1_records: Dict[str, Tuple[str, str, str]] = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        for i, row in enumerate(r):
            if not row or len(row) < 4:
                continue
            s1_records[row[0]] = (row[1], row[2], row[3])
            if sample_size > 0 and len(s1_records) >= sample_size:
                break

    # 2. Load ground truth for sample
    print("Loading Ground Truth labels...")
    ground_truth: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if not row:
                continue
            s1_id = row[0]
            if s1_id in s1_records:
                matches = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()
                ground_truth[s1_id] = matches

    all_true_target_ids: Set[str] = set()
    for matches in ground_truth.values():
        all_true_target_ids.update(matches)

    total_true_matches = len(all_true_target_ids)
    true_s2_matches = sum(1 for m in all_true_target_ids if m.startswith("S2-"))
    true_s3_matches = sum(1 for m in all_true_target_ids if m.startswith("S3-"))

    singletons_count = sum(1 for s1, m in ground_truth.items() if not m)
    print(f"Sample Ground Truth: {len(s1_records)} S1 entities ({singletons_count} singletons, {len(s1_records)-singletons_count} with matches)")
    print(f"Total True Target Links: {total_true_matches} ({true_s2_matches} S2, {true_s3_matches} S3)")

    # 3. Load target records from S2 and S3 (true matches + background target pool)
    needed_s2 = {m for m in all_true_target_ids if m.startswith("S2-")}
    needed_s3 = {m for m in all_true_target_ids if m.startswith("S3-")}

    target_pool: Dict[str, Tuple[str, str, str]] = {}
    for src_file, needed_set in [
        ("train_source2.tsv", needed_s2),
        ("train_source3.tsv", needed_s3)
    ]:
        p = os.path.join(train_dir, src_file)
        if not os.path.exists(p):
            continue
        print(f"Scanning target pool from {src_file}...")
        bg_count = 0
        with open(p, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                if not row or len(row) < 4:
                    continue
                eid = row[0]
                if eid in needed_set:
                    target_pool[eid] = (row[1], row[2], row[3])
                    needed_set.remove(eid)
                elif bg_count < background_target_sample:
                    target_pool[eid] = (row[1], row[2], row[3])
                    bg_count += 1
                if not needed_set and bg_count >= background_target_sample:
                    break

    print(f"Target pool indexed: {len(target_pool)} total records.")

    # 4. Build Country-Partitioned Multi-Strategy Blocking Indices
    print("Building country-partitioned multi-strategy indices...")
    country_indices: Dict[str, MultiStrategyBlockingIndex] = {}
    for eid, (t_name, t_addr, t_country) in target_pool.items():
        if t_country not in country_indices:
            country_indices[t_country] = MultiStrategyBlockingIndex(max_candidates=max_candidates)
        country_indices[t_country].index_target_record(eid, t_name, t_addr, t_country)

    for country, idx in country_indices.items():
        pruned = idx.prune_frequent_keys()
        print(f"  Country '{country}': {len(idx.records)} targets indexed ({pruned} high-frequency keys pruned).")

    # 5. Evaluate Candidate Generation & Measure Recall
    print("\nRunning candidate queries and tracking strategy provenance...")

    retained_true_matches = 0
    retained_s2_matches = 0
    retained_s3_matches = 0

    candidate_counts: List[int] = []
    strategy_hit_counts: Counter = Counter()
    strategy_alone_hits: Counter = Counter()
    provenance_size_dist: Counter = Counter()

    total_candidates_generated = 0

    for s1_id, (s1_name, s1_addr, s1_country) in s1_records.items():
        true_set = ground_truth.get(s1_id, set())

        index = country_indices.get(s1_country)
        if not index:
            candidate_counts.append(0)
            continue

        cand_provenance = index.query_candidates_with_provenance(
            s1_name, s1_addr, max_candidates=max_candidates
        )
        candidates = set(cand_provenance.keys())
        candidate_counts.append(len(candidates))
        total_candidates_generated += len(candidates)

        # Check hits
        hits = candidates & true_set
        retained_true_matches += len(hits)
        for h in hits:
            if h.startswith("S2-"):
                retained_s2_matches += 1
            elif h.startswith("S3-"):
                retained_s3_matches += 1

            # Track which strategies found this true match
            strats_for_hit = cand_provenance[h]
            for s in strats_for_hit:
                strategy_hit_counts[s] += 1
            if len(strats_for_hit) == 1:
                strategy_alone_hits[list(strats_for_hit)[0]] += 1
            provenance_size_dist[len(strats_for_hit)] += 1

    # 6. Calculate Metrics
    recall_total = retained_true_matches / max(total_true_matches, 1)
    recall_s2 = retained_s2_matches / max(true_s2_matches, 1)
    recall_s3 = retained_s3_matches / max(true_s3_matches, 1)

    mean_cand_count = float(np.mean(candidate_counts))
    median_cand_count = float(np.median(candidate_counts))
    max_cand_count = int(np.max(candidate_counts))
    min_cand_count = int(np.min(candidate_counts))

    # Reduction Ratio compared to Cartesian comparison with target pool
    cartesian_space = len(s1_records) * len(target_pool)
    reduction_ratio = 1.0 - (total_candidates_generated / max(cartesian_space, 1))

    dt = time.time() - t0

    # 7. Print Comprehensive Report
    print("\n" + "=" * 70)
    print("BLOCKING EVALUATION REPORT")
    print("=" * 70)
    print(f"Overall Recall (True Matches Retained):  {recall_total:.2%} ({retained_true_matches}/{total_true_matches})")
    print(f"  Source 1 -> Source 2 Recall:           {recall_s2:.2%} ({retained_s2_matches}/{true_s2_matches})")
    print(f"  Source 1 -> Source 3 Recall:           {recall_s3:.2%} ({retained_s3_matches}/{true_s3_matches})")
    print("-" * 70)
    print("Candidate Count Statistics per S1 Entity:")
    print(f"  Average Candidate Count:               {mean_cand_count:.2f}")
    print(f"  Median Candidate Count:                {median_cand_count:.1f}")
    print(f"  Min Candidate Count:                   {min_cand_count}")
    print(f"  Max Candidate Count:                   {max_cand_count}")
    print(f"  Total Candidates Generated:            {total_candidates_generated}")
    print(f"  Search Space Reduction Ratio:          {reduction_ratio:.6%}")
    print("-" * 70)
    print("Recall Breakdown by Strategy (Hits on True Matches):")
    for strat in ALL_STRATEGIES:
        hits = strategy_hit_counts.get(strat, 0)
        alone = strategy_alone_hits.get(strat, 0)
        pct = hits / max(total_true_matches, 1) * 100
        print(f"  • {strat:<28}: {hits:>5} hits ({pct:>5.1f}%) | {alone:>4} unique saves")
    print("-" * 70)
    print("Strategy Overlap Distribution for True Matches:")
    for num_strats, cnt in sorted(provenance_size_dist.items()):
        pct = cnt / max(total_true_matches, 1) * 100
        print(f"  • Found by {num_strats} strategies simultaneously: {cnt:>5} ({pct:>5.1f}%)")
    print("-" * 70)
    print(f"Evaluation finished in {dt:.1f} seconds.")
    print("=" * 70)

    results = {
        "sample_size": sample_size,
        "max_candidates_setting": max_candidates,
        "total_true_matches": total_true_matches,
        "retained_true_matches": retained_true_matches,
        "overall_recall": round(recall_total, 4),
        "recall_s2": round(recall_s2, 4),
        "recall_s3": round(recall_s3, 4),
        "mean_candidate_count": round(mean_cand_count, 2),
        "median_candidate_count": round(median_cand_count, 1),
        "max_candidate_count": max_cand_count,
        "reduction_ratio": round(reduction_ratio, 6),
        "strategy_hit_counts": dict(strategy_hit_counts),
        "strategy_unique_saves": dict(strategy_alone_hits),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Candidate Generation / Blocking Recall")
    parser.add_argument("--data-dir", default=TRAIN_DIR, help="Path to training data directory")
    parser.add_argument("--sample-size", type=int, default=3000, help="Number of S1 entities to evaluate")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES_PER_ENTITY, help="Max candidates per S1")
    parser.add_argument("--bg-sample", type=int, default=80000, help="Background target sample count")
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    results = evaluate_blocking(
        train_dir=args.data_dir,
        sample_size=args.sample_size,
        max_candidates=args.max_candidates,
        background_target_sample=args.bg_sample
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
