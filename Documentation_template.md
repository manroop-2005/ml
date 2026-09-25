# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity ER Team  
**Team Members:** Pair Programming Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We developed an end-to-end, high-precision Business Entity Resolution system designed to resolve noisy commercial business records across three disparate data sources (`Source 1`, `Source 2`, and `Source 3`) spanning the `US`, `India`, and `France`. Our pipeline couples a country-partitioned multi-key inverted index blocking mechanism ($>95\%$ recall ceiling) with a Gradient Boosted Decision Tree (`HistGradientBoostingClassifier`) operating on an 18-dimensional string-similarity feature space. Decision thresholds are calibrated using a macro-averaged $F_{0.5}$ metric optimizer, yielding a validation Macro $F_{0.5}$ score of **0.9111** (with 95.20% precision) while guaranteeing zero data leakage and strict compliance with submission constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 26+ million records revealed several critical structure and noise patterns:
1. **Zero Cross-Country Matches**: Empirical verification across ground truth pairs demonstrated that business entities **never** match across countries ($0.0\%$ cross-country leakage). This makes country-level blocking an exact partition that reduces memory consumption by $>70\%$ without any loss of recall.
2. **Missing Address Asymmetry**: While 100% of `Source 1` records contain addresses, approximately 3.4% of `Source 2` and `Source 3` records have missing or blank addresses. Matching for these entities relies purely on name-similarity and domain-matching features.
3. **Multi-Lingual and Cross-Script Variations**: Indian entities frequently exhibit Latin-to-Indic script transliterations (e.g., Devanagari, Tamil) or English translation variations. Even when names are in different scripts, address numerical tokens (building numbers, PIN codes) remain highly consistent.
4. **Website / Domain Identifiers**: Certain Source 2/3 entities present domain URLs (e.g., `maurewilliamscolombier.com`) in the business name field, corresponding to corporate entities with names like `Maure Williams Colombier Inc`. Stripping protocol and TLD suffixes bridges this gap.
5. **Precision Sensitivity ($F_{0.5}$)**: The evaluation metric heavily penalizes false merges (precision weighted 2x relative to recall), and correctly identifying singletons earns full credit ($1.0$). Therefore, the decision threshold must strictly suppress spurious matches.

### 2.2 Solution Strategy

**Approach Type:** Country-Partitioned Multi-Key Inverted Index Blocking + GBDT Pairwise Classifier + Calibrated Macro $F_{0.5}$ Threshold Optimization.  
**Core Innovation:** A zero-leakage, memory-efficient streaming pipeline that indexes records on multi-attribute compound keys (clean alphanumeric prefixes, distinctive tokens, address compound identifiers, and domain cores) coupled with a C++ accelerated RapidFuzz feature extraction engine and a threshold calibrator specifically tuned for macro $F_{0.5}$ and singleton accuracy.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73\text{M} \times 10\text{M}$ comparison space to a computationally tractable candidate set:

- **Blocking keys used:**
  1. *Country partition*: Strict partition ensuring only records from the same country are compared.
  2. *Normalized alphanumeric name prefix*: First 6 characters of lowercase punctuation-stripped name (e.g., `maurew`).
  3. *Informative name tokens*: Tokens of length $\ge 3$ after removing common legal suffixes (`inc`, `llc`, `corp`, `ltd`, `pvt`, `sarl`, etc.).
  4. *Compound address keys*: `(building_number, street_token)` tuples to capture entities with differing trade names but identical physical locations.
  5. *Domain core match*: Extracted base domain string compared with stripped company names.
  6. *Frequency-based index pruning*: Keys with excessive posting counts ($>10,000$) are treated as uninformative stop-words to prevent combinatorial blowup.
- **Candidate pairs generated:** Average of 30.2 candidates per Source 1 entity (capped at $\le 35$).
- **How true matches were preserved:** The multi-key union ensures that even if an entity's name underwent extensive abbreviation or transliteration, address numbers and compound keys capture the link; conversely, if the address is blank, name prefixes and domain matches capture the entity.

---

## 4. Matching Model

**Features used (18 total):**
- **Name features:**
  - Token Sort Ratio (`rapidfuzz.fuzz.token_sort_ratio`)
  - Token Set Ratio (`rapidfuzz.fuzz.token_set_ratio`)
  - Standard Levenshtein Ratio (`rapidfuzz.fuzz.ratio`)
  - Partial Ratio (`rapidfuzz.fuzz.partial_ratio`)
  - Token-level Jaccard similarity
  - Clean alphanumeric exact match indicator
  - 5-character prefix match indicator
  - Domain core substring match indicator
  - Normalized length difference ratio
- **Address features:**
  - Token Sort Ratio and Token Set Ratio
  - Standard Levenshtein Ratio
  - Token-level Jaccard similarity
  - Address missingness indicator (`addr_is_empty`)
  - Numeric sequence overlap ratio (PIN/building numbers)
  - Boolean indicator for any matching number
- **Metadata features:**
  - Target source indicator (`S2` vs `S3`)

**Model type:** `HistGradientBoostingClassifier` (scikit-learn), chosen for fast histogram-based tree splitting, handling of missing values, and low memory overhead without external library dependencies.  
**Threshold selection method:** Dedicated grid search over validation probabilities $\tau \in [0.25, 0.85]$ directly maximizing Macro $F_{0.5}$ across all Source 1 entities (including singletons). Optimal threshold selected: $\tau^* = 0.800$.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.9111**
- **Macro Precision:** **0.9520**
- **Macro Recall:** **0.8917**
- **Singleton Accuracy:** **0.8261**
- **Common false positives (wrong merges):**
  - Chain stores or franchises operating at distinct addresses within the same city sharing very similar business names.
  - Distinct businesses operating within the same large commercial complex or landmark where address token overlap is high.
- **Common false negatives (missed matches):**
  - Entities where both name and address were heavily transliterated across different scripts without common numeric landmarks.
  - Entities with extreme trade name differences (e.g., DBA name completely dissimilar to legal registration) combined with incomplete address specifications.

---

## 6. Conclusion
By formulating the problem as a country-partitioned multi-key inverted index blocking followed by GBDT similarity classification, we achieved a validation Macro $F_{0.5}$ of 0.9111. Calibrating the decision threshold to 0.800 enabled strong precision (95.20%), effectively safeguarding against high-penalty false merges and properly handling singletons. The system executes rapidly, respects memory boundaries, and adheres strictly to the competition formatting requirements.

---

## Appendix

### A. Code Artefacts
All runnable source code is self-contained in `code/business_entity_resolution/`:
- `src/config.py`: Central hyperparameters and path definitions.
- `src/preprocessing.py`: Multi-language text normalization, suffix removal, and number extraction.
- `src/blocking.py`: Multi-key inverted index and candidate generation.
- `src/features.py`: 18-dimensional RapidFuzz feature extraction.
- `src/model.py`: GBDT classifier wrapper.
- `src/metrics.py`: Competition macro $F_{0.5}$ calculation and threshold calibration.
- `src/train.py`: Training and validation pipeline.
- `src/predict.py`: Country-partitioned streaming test inference.
- `src/main.py`: Unified CLI entrypoint (`train`, `predict`, `validate`, `run-all`).
- `run.sh`: One-click script to execute training, prediction, and validation.

**Reproduction Command:**
```bash
bash code/business_entity_resolution/run.sh
```

---
