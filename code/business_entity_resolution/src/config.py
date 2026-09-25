"""
Global configuration and hyperparameters for Business Entity Resolution.
"""

import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = os.environ.get("DATA_DIR", str(ROOT_DIR / "dataset"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Model and Output paths
DEFAULT_MODEL_DIR = str(ROOT_DIR / "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "entity_resolution_model.joblib")
DEFAULT_OUTPUT_DIR = str(ROOT_DIR / "output")
MATCHING_RESULTS_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "candidate_pairs.tsv")

# Country labels (open set support - pipeline dynamically handles any country label)
KNOWN_COUNTRIES = ("US", "India", "France")

# Blocking / Candidate Generation Hyperparameters
MAX_CANDIDATES_PER_ENTITY = 35
MIN_TOKEN_LEN = 3
MIN_ADDR_NUM_LEN = 2
MAX_BLOCK_POSTINGS = 250  # prune index terms appearing in > 250 postings

# Common stop words and legal entity suffixes across US, India, and France
LEGAL_SUFFIXES = {
    # US / General English
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "co", "company", "enterprises", "solutions", "group", "services", "tech",
    "technologies", "holdings", "partners", "assoc", "associates", "lp", "llp",
    "pllc", "pc", "pa", "foundation", "institute", "center", "centre", "club",
    # India
    "pvt", "private", "pte", "m/s", "ms",
    # France
    "sa", "sarl", "sas", "sasu", "snc", "sci", "sca", "eurl", "gie"
}

ADDRESS_STOPWORDS = {
    # English
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "parkway", "pkwy", "highway", "hwy",
    "way", "circle", "cir", "suite", "ste", "apartment", "apt", "floor", "fl",
    "building", "bldg", "unit", "block", "blk", "sector", "sec", "phase",
    "near", "opp", "opposite", "behind", "beside", "at", "post", "po", "box",
    "north", "south", "east", "west", "n", "s", "e", "w",
    # Indian context
    "nagar", "colony", "marg", "chowk", "bazar", "bazaar", "gali", "mohalla",
    "taluk", "dist", "district", "pradesh",
    # French context
    "rue", "avenue", "boulevard", "place", "route", "chemin", "allee", "impasse",
    "cours", "quai", "passage", "square", "cedex", "bp", "porte", "batiment",
    # Common function words
    "the", "and", "of", "for", "in", "on", "de", "du", "des", "la", "le", "les", "et"
}

# Machine Learning Classifier Parameters
MODEL_PARAMS = {
    "learning_rate": 0.07,
    "max_iter": 350,
    "max_leaf_nodes": 47,
    "min_samples_leaf": 15,
    "l2_regularization": 0.5,
    "class_weight": "balanced",
    "random_state": 42
}

# Metric Parameters
BETA = 0.5  # Precision-heavy F_0.5
DEFAULT_THRESHOLD = 0.55  # Default probability cutoff before threshold tuning

# Threshold grid for grid search optimization (finer granularity = better calibration)
THRESHOLD_GRID = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]
