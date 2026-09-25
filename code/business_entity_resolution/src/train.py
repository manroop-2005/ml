"""
Training pipeline for Business Entity Resolution.
Generates candidates, builds feature dataset, trains GBDT, optimizes F_0.5 threshold,
and saves the trained model artifact.
"""

import os
import csv
import sys
import time
import random
import argparse
from typing import Dict, List, Set, Tuple
import numpy as np

from .config import (
    TRAIN_DIR,
    DEFAULT_MODEL_PATH,
    KNOWN_COUNTRIES,
    BETA,
    THRESHOLD_GRID,
)
from .preprocessing import normalize_text
from .blocking import BlockingIndex
from .features import compute_pair_features, FEATURE_NAMES
from .model import EntityResolutionModel
from .metrics import compute_macro_f_beta, find_optimal_threshold


def load_s1_and_ground_truth(
    train_dir: str,
    sample_size: int = 30000,
    seed: int = 42
) -> Tuple[Dict[str, Tuple[str, str, str]], Dict[str, Set[str]]]:
    """Load Source 1 records and ground truth labels, optionally subsampled."""
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"Loading Source 1 records from {s1_path}...")
    s1_records: Dict[str, Tuple[str, str, str]] = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for i, row in enumerate(reader):
            if not row or len(row) < 4:
                continue
            s1_records[row[0]] = (row[1], row[2], row[3])
            if sample_size > 0 and len(s1_records) >= sample_size:
                break

    print(f"Loaded {len(s1_records)} Source 1 records.")

    print(f"Loading Ground Truth from {gt_path}...")
    ground_truth: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            if s1_id in s1_records:
                matches = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()
                ground_truth[s1_id] = matches

    # Fill in singletons with empty set if missing from ground truth file
    for s1_id in s1_records:
        if s1_id not in ground_truth:
            ground_truth[s1_id] = set()

    non_empty = sum(1 for m in ground_truth.values() if m)
    print(f"Ground truth loaded: {len(ground_truth)} records ({non_empty} with matches, {len(ground_truth) - non_empty} singletons).")
    return s1_records, ground_truth


def load_target_pool(
    train_dir: str,
    needed_ids: Set[str],
    target_sample_limit: int = 150000
) -> Dict[str, Tuple[str, str, str]]:
    """
    Load target records from Source 2 and Source 3.
    Ensures all needed ground-truth target IDs are loaded, along with a background sample of records.
    """
    targets: Dict[str, Tuple[str, str, str]] = {}
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    needed_by_source = {
        s2_path: {eid for eid in needed_ids if eid.startswith("S2-")},
        s3_path: {eid for eid in needed_ids if eid.startswith("S3-")}
    }

    for path, remaining_needed in needed_by_source.items():
        if not os.path.exists(path):
            continue
        print(f"Scanning target records from {os.path.basename(path)} (need {len(remaining_needed)} matches)...")
        bg_count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                eid = row[0]
                is_needed = eid in remaining_needed
                if is_needed:
                    targets[eid] = (row[1], row[2], row[3])
                    remaining_needed.remove(eid)
                elif target_sample_limit > 0 and bg_count < target_sample_limit:
                    targets[eid] = (row[1], row[2], row[3])
                    bg_count += 1

                if target_sample_limit > 0 and not remaining_needed and bg_count >= target_sample_limit:
                    break

    found_needed = sum(1 for tid in needed_ids if tid in targets)
    print(f"Total target records indexed: {len(targets)} (covered {found_needed}/{len(needed_ids)} ground-truth matches).")
    return targets


def train_pipeline(
    train_dir: str = TRAIN_DIR,
    sample_size: int = 30000,
    val_ratio: float = 0.15,
    model_out: str = DEFAULT_MODEL_PATH,
    seed: int = 42
):
    """Run end-to-end training and validation."""
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("Starting ML Challenge 2026 Entity Resolution Training")
    print(f"Data directory: {train_dir}")
    print(f"Sample size: {sample_size} (0 = full)")
    print(f"Validation ratio: {val_ratio}")
    print(f"Output model path: {model_out}")
    print("=" * 60)

    # 1. Load S1 and Ground Truth
    t0 = time.time()
    s1_records, ground_truth = load_s1_and_ground_truth(train_dir, sample_size=sample_size, seed=seed)

    # 2. Train / Val split
    s1_ids = list(s1_records.keys())
    random.shuffle(s1_ids)
    val_split_idx = int(len(s1_ids) * (1.0 - val_ratio))
    train_s1_ids = set(s1_ids[:val_split_idx])
    val_s1_ids = set(s1_ids[val_split_idx:])
    print(f"Split: {len(train_s1_ids)} training entities, {len(val_s1_ids)} validation entities.")

    # 3. Identify all true match IDs needed
    needed_target_ids: Set[str] = set()
    for s1_id in s1_records:
        needed_target_ids.update(ground_truth.get(s1_id, set()))

    # 4. Load Target Pool (S2 and S3)
    target_sample_limit = 200000 if sample_size > 0 else 0
    targets = load_target_pool(train_dir, needed_target_ids, target_sample_limit=target_sample_limit)

    # 5. Build country-partitioned blocking indices
    print("Building country-partitioned blocking indices...")
    country_indices: Dict[str, BlockingIndex] = {}
    for eid, (t_name, t_addr, t_country) in targets.items():
        if t_country not in country_indices:
            country_indices[t_country] = BlockingIndex()
        country_indices[t_country].index_target_record(eid, t_name, t_addr, t_country)

    for country, index in country_indices.items():
        index.prune_frequent_keys()
        print(f"  Country '{country}': {len(index.records)} target records indexed.")

    # 6. Candidate Generation & Feature Extraction for Training set
    print("Generating candidate pairs and extracting features for training...")
    X_train_list: List[List[float]] = []
    y_train_list: List[int] = []

    pos_count = 0
    neg_count = 0
    # Use hard negatives: sample from retrieved candidates (not random universe)
    # Max negatives = 10x positives per entity to maintain signal quality
    MAX_NEG_PER_ENTITY = 10

    for s1_id in train_s1_ids:
        s1_name, s1_addr, s1_country = s1_records[s1_id]
        true_matches = ground_truth.get(s1_id, set())

        index = country_indices.get(s1_country)
        candidates = index.query_candidates(s1_name, s1_addr) if index else []

        # Always include true matches in candidate set during training to learn positive features
        cand_set = set(candidates) | (true_matches & targets.keys())

        # Hard negatives: sample from actual retrieved candidates (not true matches)
        # These are confusable candidates the model must learn to reject
        neg_candidates = [cid for cid in candidates if cid not in true_matches]
        if len(neg_candidates) > MAX_NEG_PER_ENTITY:
            neg_candidates = random.sample(neg_candidates, MAX_NEG_PER_ENTITY)

        eval_cands = (true_matches & targets.keys()) | set(neg_candidates)

        for cid in eval_cands:
            if cid not in targets:
                continue
            cand_name, cand_addr, _ = targets[cid]
            feats = compute_pair_features(s1_name, s1_addr, cid, cand_name, cand_addr)
            label = 1 if cid in true_matches else 0

            X_train_list.append(feats)
            y_train_list.append(label)

            if label == 1:
                pos_count += 1
            else:
                neg_count += 1

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"Training feature matrix shape: {X_train.shape} ({pos_count} positives, {neg_count} negatives)")

    # 7. Train GBDT Classifier
    print("Training Gradient Boosted Decision Tree (HistGradientBoostingClassifier)...")
    clf_model = EntityResolutionModel()
    clf_model.fit(X_train, y_train)

    # 8. Evaluate on Validation Set and Optimize Threshold
    print("Evaluating on validation set and optimizing Macro F_0.5 threshold...")
    val_cand_scores: Dict[str, List[Tuple[str, float]]] = {}
    val_gt = {s1_id: ground_truth[s1_id] for s1_id in val_s1_ids}

    val_cand_pairs_count = 0
    for s1_id in val_s1_ids:
        s1_name, s1_addr, s1_country = s1_records[s1_id]
        index = country_indices.get(s1_country)
        candidates = index.query_candidates(s1_name, s1_addr) if index else []

        s1_scores: List[Tuple[str, float]] = []
        if candidates:
            cand_features = []
            valid_cands = []
            for cid in candidates:
                if cid not in targets:
                    continue
                cand_name, cand_addr, _ = targets[cid]
                feats = compute_pair_features(s1_name, s1_addr, cid, cand_name, cand_addr)
                cand_features.append(feats)
                valid_cands.append(cid)

            if cand_features:
                X_val = np.array(cand_features, dtype=np.float32)
                probs = clf_model.predict_proba(X_val)
                s1_scores = list(zip(valid_cands, probs))
                val_cand_pairs_count += len(s1_scores)

        val_cand_scores[s1_id] = s1_scores

    # Optimize threshold using the finer THRESHOLD_GRID from config
    best_thresh, best_f05 = find_optimal_threshold(
        val_gt, val_cand_scores, threshold_grid=THRESHOLD_GRID, beta=BETA
    )
    clf_model.threshold = best_thresh

    # Compute final validation metrics at optimal threshold
    val_preds: Dict[str, Set[str]] = {}
    for s1_id, cands in val_cand_scores.items():
        val_preds[s1_id] = {cid for cid, score in cands if score >= best_thresh}

    macro_f05, macro_p, macro_r = compute_macro_f_beta(val_gt, val_preds, beta=BETA)

    # Singleton specific evaluation
    singleton_ids = [s1 for s1, true_m in val_gt.items() if not true_m]
    if singleton_ids:
        correct_singletons = sum(1 for s1 in singleton_ids if not val_preds[s1])
        singleton_acc = correct_singletons / len(singleton_ids)
    else:
        singleton_acc = 1.0

    print("-" * 60)
    print("Validation Results:")
    print(f"  Optimal Probability Threshold: {best_thresh:.3f}")
    print(f"  Macro F_0.5 Score:             {macro_f05:.4f}")
    print(f"  Macro Precision:               {macro_p:.4f}")
    print(f"  Macro Recall:                  {macro_r:.4f}")
    print(f"  Singleton Accuracy:            {singleton_acc:.4f} ({len(singleton_ids)} total singletons)")
    print(f"  Total validation pairs scored: {val_cand_pairs_count}")
    print("-" * 60)

    # 9. Save model artifact
    print(f"Saving model artifact to {model_out}...")
    clf_model.save(model_out)
    dt = time.time() - t0
    print(f"Training completed successfully in {dt:.1f} seconds.")
    print("=" * 60)
    return clf_model, macro_f05


def main():
    parser = argparse.ArgumentParser(description="Train Entity Resolution GBDT Model")
    parser.add_argument("--data-dir", default=TRAIN_DIR, help="Path to training data directory")
    parser.add_argument("--sample-size", type=int, default=25000, help="Number of S1 entities to sample (0 = all)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Output path for trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_pipeline(
        train_dir=args.data_dir,
        sample_size=args.sample_size,
        val_ratio=args.val_ratio,
        model_out=args.model_out,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
