# Exploratory Data Analysis Report: Business Entity Resolution

**Challenge:** ML Challenge 2026 — Business Entity Resolution  
**Total Records Analyzed:** 26,435,994 rows across 7 files  
**Analysis Date:** September 2026  

---

## 1. Executive Summary

This exploratory data analysis covers all 7 files provided in the Business Entity Resolution dataset across training and test splits. The goal of the challenge is to match records from two secondary sources (`Source 2` and `Source 3`) to a deduplicated reference source (`Source 1`).

### High-Level Dataset Inventory

| File | Type | File Size | Total Rows | Columns | Unique Entity IDs | Missing Addresses (%) |
|---|---|---|---|---|---|---|
| **`train_source1.tsv`** | Train Reference | 200.34 MB | 2,206,821 | 4 | 2,206,821 | 0.00% (None) |
| **`train_source2.tsv`** | Train Secondary | 466.63 MB | 5,034,616 | 4 | 5,034,616 | 3.36% (168,967) |
| **`train_source3.tsv`** | Train Secondary | 480.37 MB | 5,285,603 | 4 | 5,285,603 | 3.33% (175,916) |
| **`train_ground_truth.tsv`** | Train Labels | 121.13 MB | 2,206,821 | 2 | 2,206,821 | N/A (5.58% singletons) |
| **`test_source1.tsv`** | Test Reference | 166.91 MB | 1,732,544 | 4 | 1,732,544 | 0.00% (None) |
| **`test_source2.tsv`** | Test Secondary | 485.86 MB | 4,887,273 | 4 | 4,887,273 | 2.65% (129,408) |
| **`test_source3.tsv`** | Test Secondary | 482.56 MB | 5,082,316 | 4 | 5,082,316 | 2.68% (136,098) |

**Key Findings:**
- **Zero Duplicate IDs**: All `entity_id` values across every file are strictly unique (0 duplicates).
- **Exact Country Alignment**: Matches never cross national boundaries (empirical check confirmed 0 cross-country matches).
- **Country Distribution Shift**: Training data contains only `US` and `India`, whereas test data introduces a third country, `France` (~14.5% to 15.0% of test records).
- **Asymmetric Missing Data**: `Source 1` has zero missing values in both train and test. In contrast, `Source 2` and `Source 3` have ~2.6% to 3.4% missing addresses. Names and country labels are 100% complete across all files.

---

## 2. File-by-File Breakdown

### 2.1 `dataset/train/train_source1.tsv`
The deduplicated reference training file. Every entity here is an anchor to be resolved.

1. **Number of rows:** 2,206,821
2. **Column names:** `entity_id`, `business_name`, `business_address`, `country`
3. **Data types:** String / text (tab-separated)
4. **Missing values:** 0 across all columns (100% complete)
5. **Duplicate entity IDs:** 0
6. **Unique entity IDs:** 2,206,821
7. **Country distribution:**
   - `US`: 1,323,633 (59.98%)
   - `India`: 883,188 (40.02%)
8. **Examples of business names:**
   - `"Orelee's Barbershop"`
   - `"Prime Money"`
   - `"B+ Retail Inc"`
   - `"Sunrise Shree Ventures Private Limited"`
   - `"Ambika Infotech Private Limited"`
   - `"Total Chemical Innovations, LLC"`
9. **Examples of business addresses:**
   - `"1795 Westchester Drive, High Point, NC"`
   - `"17560 Ellis Road, Tahlequah, OK"`
   - `"P No. 46, Block-A, Tagore Circle, Jaipur, Rajasthan"`
   - `"No.3, Wood Street, Bangalore, Karnataka"`
   - `"19-51 37 Street, Astoria, NY"`
10. **Observed patterns:** Canonical business names with clean standard punctuation; addresses contain complete street, city, state, or PIN components.

---

### 2.2 `dataset/train/train_source2.tsv`
First noisy secondary training source.

1. **Number of rows:** 5,034,616
2. **Column names:** `entity_id`, `business_name`, `business_address`, `country`
3. **Data types:** String / text (tab-separated)
4. **Missing values per column:**
   - `entity_id`: 0 (0.0%)
   - `business_name`: 0 (0.0%)
   - `business_address`: 168,967 (3.36%)
   - `country`: 0 (0.0%)
5. **Duplicate entity IDs:** 0
6. **Unique entity IDs:** 5,034,616
7. **Country distribution:**
   - `US`: 3,016,817 (59.92%)
   - `India`: 2,017,799 (40.08%)
8. **Examples of business names:**
   - `"-- Holloway Peak Inc Seafood"` (leading punctuation noise)
   - `"राम मार्केटिंग प्राइवेट लिमिटेड"` (Devanagari script translation)
   - `"आदित्य प्रॉपर्टीज एलएलपी"` (Devanagari script translation of LLP)
   - `"Group Lio Harbor Ímaging"` (accented diacritic)
   - `"Rapid Institute of Teclnholeoeg"` (typo in "Technology")
   - `"Supreme  Investment Corp Services"` (double spaces and suffix combination)
9. **Examples of business addresses:**
   - `"105 ELM ST, MORGANTON, NC"` (all caps)
   - `"KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"` (Khasra numbering)
   - `"##220 DYER AVENUE, COOKEVILLE, TN"` (symbol noise `##`)
   - `"TX, MCKINNEY, 2511 CLEAR BROOK DR"` (inverted state-first format)
   - `"2/16, RAGHUPATHY LAYOUT SAIBABA COLONY, தமிழ்நாடு"` (Tamil script in state name)
10. **Observed patterns:** 
    - Severe case inconsistency (heavy uppercase usage).
    - Missing addresses (3.36%) requiring name-only matching.
    - Script variations (Devanagari, Tamil) for Indian entities.

---

### 2.3 `dataset/train/train_source3.tsv`
Second noisy secondary training source.

1. **Number of rows:** 5,285,603
2. **Column names:** `entity_id`, `business_name`, `business_address`, `country`
3. **Data types:** String / text (tab-separated)
4. **Missing values per column:**
   - `entity_id`: 0 (0.0%)
   - `business_name`: 0 (0.0%)
   - `business_address`: 175,916 (3.33%)
   - `country`: 0 (0.0%)
5. **Duplicate entity IDs:** 0
6. **Unique entity IDs:** 5,285,603
7. **Country distribution:**
   - `US`: 3,170,056 (59.98%)
   - `India`: 2,115,547 (40.02%)
8. **Examples of business names:**
   - `"wilfordhancock.com"` (website URL format instead of company name)
   - `"LLC Moncada Léarning Center"` (legal prefix placed before name)
   - `"Rosette Trafel Trafel Limited"` (repeated token typo)
   - `"8asheerbagh Busilnedss Private Limited"` (digit-prefixed typo)
   - `"Maid Náils"` (accented character)
9. **Examples of business addresses:**
   - `"Mack Rd, Haltom City, Texas"` (abbreviated Road, full state name)
   - `"#2069 San Bernardino Avenue, # UNIT 3109, Colton, California"`
   - `"7485 1/2 Woodlake Drive, PMB 6094, Walton Hills, Ohio"` (fractional street number and PMB)
   - `"తెలంగాణ, H.no 542 Suite No.14, Basheerbagh, Hyderabad-1."` (Telugu script state prefix)
   - `"S 22 Jai Ambe Complex, Nagpur, MH"` (state code abbreviation)
10. **Observed patterns:**
    - Domain names appearing as company names.
    - Inverted legal suffixes (`LLC Moncada...`).
    - Multi-lingual address components.
    - ~3.33% missing addresses.

---

### 2.4 `dataset/train/train_ground_truth.tsv`
Ground truth mapping from `Source 1` reference IDs to matching `Source 2` and `Source 3` IDs.

1. **Number of rows:** 2,206,821 (exactly matches `train_source1.tsv`)
2. **Column names:** `source1_entity_id`, `matched_entity_ids`
3. **Data types:** String / text (tab-separated, matches comma-separated)
4. **Missing values per column:**
   - `source1_entity_id`: 0 (0.0%)
   - `matched_entity_ids`: 123,247 (5.58% — true singletons with no matches)
5. **Duplicate entity IDs:** 0
6. **Unique entity IDs:** 2,206,821
7. **Matches summary:**
   - Total matched references: 7,638,365
   - Matches from `Source 2`: 3,693,619 (48.36%)
   - Matches from `Source 3`: 3,944,746 (51.64%)

---

### 2.5 `dataset/test/test_source1.tsv`
Reference test file. Every entity must be present in the submission output files.

1. **Number of rows:** 1,732,544
2. **Column names:** `entity_id`, `business_name`, `business_address`, `country`
3. **Data types:** String / text (tab-separated)
4. **Missing values:** 0 across all columns (100% complete)
5. **Duplicate entity IDs:** 0
6. **Unique entity IDs:** 1,732,544
7. **Country distribution:**
   - `India`: 809,986 (46.75%)
   - `US`: 663,106 (38.27%)
   - `France`: 259,452 (14.98%) — *New country not present in training data*
8. **Examples of business names:**
   - `"Zephay Labs Inc"` (US)
   - `"Vision Partners Corp"` (US)
   - `"<< Team Ecole"` (France — leading symbol noise)
   - `"Classe & Cie SARL"` (France — French legal entity suffix SARL)
   - `"Great Management Private Limited"` (India)
   - `"QC Services Pvt Ltd"` (India)
9. **Examples of business addresses:**
   - `"2621 Cotten Road, Tyler, TX"`
   - `"175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine"`
   - `"14 Rue Royer, Dunkerque, Hauts-de-France"`
   - `"F-2/33, Village Chakkarpur, Dlf, Phase-1 Tehsil Wazirabad, Dlf Qe, Gurgaon, Haryana"`
10. **Observed patterns:** Introduces French addresses (`Boulevard`, `Rue`, `Hauts-de-France`, `Nouvelle-Aquitaine`) and legal structures (`SARL`, `SAS`, `Cie`).

---

### 2.6 `dataset/test/test_source2.tsv`
First test secondary source.

1. **Number of rows:** 4,887,273
2. **Column names:** `entity_id`, `business_name`, `business_address`, `country`
3. **Data types:** String / text (tab-separated)
4. **Missing values per column:**
   - `entity_id`: 0 (0.0%)
   - `business_name`: 0 (0.0%)
   - `business_address`: 129,408 (2.65%)
   - `country`: 0 (0.0%)
5. **Duplicate entity IDs:** 0
6. **Unique entity IDs:** 4,887,273
7. **Country distribution:**
   - `India`: 2,312,565 (47.32%)
   - `US`: 1,871,330 (38.29%)
   - `France`: 703,378 (14.39%)
8. **Examples of business names:**
   - `"Marina Ecole France Sarl"`
   - `"SCI Ptit Àmicale"` (French SCI real estate entity with accents)
   - `"LLC MILLER CRURECNCYHSARLESH,"` (caps with typos and trailing comma)
   - `"Clinique Bôw"`
   - `"lorraine service sci"` (all lowercase)
9. **Examples of business addresses:**
   - `"63 R. DE DIEPPE, LILLE, Hauts-de-France"`
   - `"ALL DES CEDRES, LA BAULE ESCOUBLAC, Pays de la Loire"`
   - `"N° 81 R DE RONCQ, TOURCOING, Nord"`
   - `"105 CABLE ROAD, PORTLAND, TN"`
10. **Observed patterns:** 2.65% missing addresses; French street prefixes (`R.`, `ALL`, `RUE`, `N°`).

---

### 2.7 `dataset/test/test_source3.tsv`
Second test secondary source.

1. **Number of rows:** 5,082,316
2. **Column names:** `entity_id`, `business_name`, `business_address`, `country`
3. **Data types:** String / text (tab-separated)
4. **Missing values per column:**
   - `entity_id`: 0 (0.0%)
   - `business_name`: 0 (0.0%)
   - `business_address`: 136,098 (2.68%)
   - `country`: 0 (0.0%)
5. **Duplicate entity IDs:** 0
6. **Unique entity IDs:** 5,082,316
7. **Country distribution:**
   - `India`: 2,405,000 (47.32%)
   - `US`: 1,945,701 (38.28%)
   - `France`: 731,615 (14.40%)
8. **Examples of business names:**
   - `"मॉडर्न फाइनेंस"` (Devanagari script)
   - `"Fractales Amis Groupe S.A.S"` (French SAS with punctuation)
   - `"Raymond & Partners Enterprises"`
   - `"Artistique Comite Sas"`
9. **Examples of business addresses:**
   - `"23 Rue Icmre, La Teste-de-buch, Gironde"`
   - `"56 R De Coulmiers, Nantes"`
   - `"No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR"`
   - `"69 Comanche Ln, Los Lunas, New Mexico"`
10. **Observed patterns:** 2.68% missing addresses; French communes and departments; Indic script entity names.

---

## 3. Ground Truth Deep Dive

An exhaustive analysis of all 2,206,821 ground truth records in `train_ground_truth.tsv` reveals the exact matching relationships:

### 3.1 Relationship Between Source 1 and Source 2 / Source 3 IDs

- Every row corresponds to exactly one `Source 1` reference entity (`S1-*`).
- Matched IDs consist strictly of `Source 2` (`S2-*`) and `Source 3` (`S3-*`) identifiers.
- **Cross-Source Match Proportions:**
  - Total links to `Source 2`: **3,693,619** (48.36%)
  - Total links to `Source 3`: **3,944,746** (51.64%)
  - Matches are almost evenly split between `Source 2` and `Source 3`.
- **Co-occurrence of Sources:**
  - `Source 1` entities matching **both** S2 and S3: **1,776,047** (**80.48%** of all S1 records)
  - `Source 1` entities matching **S2 only**: **143,029** (6.48%)
  - `Source 1` entities matching **S3 only**: **164,498** (7.45%)
  - `Source 1` entities matching **neither** (singletons): **123,247** (5.58%)

### 3.2 Breakdown by Match Counts

| Category | Description | Count | Percentage |
|---|---|---|---|
| **Zero Matches (Singletons)** | Entities with empty `matched_entity_ids` | **123,247** | **5.58%** |
| **Exactly One Match** | Matches exactly 1 record from S2 or S3 | **119,157** | **5.40%** |
| **Multiple Matches** | Matches 2 or more records across S2/S3 | **1,964,417** | **89.02%** |
| **Total S1 Records** | All evaluated entities | **2,206,821** | **100.00%** |

### 3.3 Full Match Count Distribution

```
Number of Matches | Count of S1 Entities | Percentage | Cumulative %
------------------+----------------------+------------+-------------
        0         |       123,247        |    5.58%   |     5.58%
        1         |       119,157        |    5.40%   |    10.98%
        2         |       375,212        |   17.00%   |    27.99%
        3         |       530,841        |   24.05%   |    52.04%
        4         |       484,115        |   21.94%   |    73.98%
        5         |       321,957        |   14.59%   |    88.57%
        6         |       164,868        |    7.47%   |    96.04%
        7         |        63,968        |    2.90%   |    98.94%
        8         |        18,680        |    0.85%   |    99.78%
        9         |         4,205        |    0.19%   |    99.97%
       10         |           534        |    0.02%   |    99.998%
       11         |            37        |    0.002%  |   100.00%
```

- **Median matches per non-empty S1 entity:** 3 matches
- **Mean matches per non-empty S1 entity:** 3.67 matches
- **Maximum matches observed for a single S1 entity:** 11 matches
- **84.5% of entities** have between 2 and 5 matches.

---

## 4. Synthesis of Noise Patterns

| Noise Category | Observed Manifestations | Practical Handling Strategy |
|---|---|---|
| **Punctuation & Symbols** | Leading `--`, `<<`, `##`, trailing commas `,` | Strip non-alphanumeric noise during tokenization. |
| **Legal Suffix Inconsistencies** | `Inc`, `LLC`, `Corp`, `Pvt Ltd`, `SARL`, `SAS` | Remove or downweight legal entity tokens in similarity scoring. |
| **Web Domains as Names** | `wilfordhancock.com`, `maurewilliamscolombier.com` | Detect domain suffixes (`.com`, `.in`, `.fr`), strip TLDs, and extract core string. |
| **Casing Variations** | ALL CAPS (`105 ELM ST`), all lowercase (`lorraine service sci`) | Casefold all strings to lowercase before comparison. |
| **Transliteration & Multi-Script** | Devanagari (`राम`), Tamil (`தமிழ்நாடு`), Telugu (`తెలంగాణ`) | Multi-lingual unicode normalization (`NFKD`); fall back to address digits. |
| **Missing Addresses** | 3.3% in train S2/S3, 2.65% in test S2/S3 are blank | Explicit `addr_is_empty` binary feature; rely on high name-similarity threshold. |
| **Address Inversions & Landmark Noise** | `TX, MCKINNEY, 2511 CLEAR BROOK DR`, `Near SBI ATM` | Use token sort ratio, token set ratio, and extract numerical components. |

---

## 5. Key Takeaways for Modeling

1. **Country Partitioning is Lossless:** Because zero matches cross national boundaries, candidate generation must be executed strictly within each country partition (`US`, `India`, `France`). This cuts search complexity and memory usage drastically.
2. **Open-Set Country Support:** France is absent from the training set but comprises 15% of the test set. Models must not hardcode country one-hot vectors and instead learn general linguistic and string-distance rules.
3. **Precision Optimization ($F_{0.5}$):** With $\beta=0.5$, precision is weighted twice as heavily as recall. False positives severely hurt the score. The probability decision threshold must be calibrated high ($\sim 0.75 - 0.85$).
4. **Singletons Matter:** 5.58% of entities have zero matches. Correctly predicting an empty list gives a score of 1.0, while predicting even a single false match on a singleton drops the score to 0.0.
