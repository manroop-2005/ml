"""
Preprocessing and normalization module for business names and addresses.

Requirements satisfied:
1. Preserves original columns (raw values remain intact).
2. Creates separate normalized columns (normalized_name, normalized_address, clean_name, address_numbers).
3. Normalizes case (lowercasing).
4. Normalizes Unicode (NFKD decomposition for Latin diacritics while preserving Indic scripts).
5. Normalizes whitespace (collapsing multiple spaces, stripping leading/trailing whitespace).
6. Handles punctuation carefully (& -> 'and', @ -> 'at', stripping extraneous noise symbols).
7. Normalizes formatting variations (domain names, leading dashes/hashes/symbols).
8. Handles business suffixes and road abbreviations based on training data analysis.
9. Preserves all numbers in addresses (vital spatial anchors).
10. Preserves all meaningful address locality, city, state, and landmark information.
11. Deterministic and reproducible (pure functions, idempotent).
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

try:
    from .data_loader import EntityRecord
except ImportError:
    from data_loader import EntityRecord

# Precompiled Regex Patterns
RE_NON_ALPHANUM_UNICODE = re.compile(r"[^\w\s\u0900-\u0D7F]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_DOMAIN_SUFFIX = re.compile(r"\.(com|in|fr|org|net|co|io|biz|info|us|gov|edu)\b", re.IGNORECASE)
RE_WWW_PREFIX = re.compile(r"^www\.", re.IGNORECASE)
RE_LEADING_NOISE = re.compile(r"^[^\w\u0900-\u0D7F]+", re.UNICODE)
RE_TRAILING_NOISE = re.compile(r"[^\w\u0900-\u0D7F]+$", re.UNICODE)

# Common business suffixes to standardize (based on empirical training data analysis)
BUSINESS_SUFFIX_RULES: List[Tuple[re.Pattern, str]] = [
    # Indian / Commonwealth
    (re.compile(r"\bpvt\.?\s*ltd\.?\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\bprivate\s+limited\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\bprivate\b", re.IGNORECASE), "pvt"),
    (re.compile(r"\blimited\b", re.IGNORECASE), "ltd"),
    (re.compile(r"\bltd\.\b", re.IGNORECASE), "ltd"),
    # US / General Corporate
    (re.compile(r"\bincorporated\b", re.IGNORECASE), "inc"),
    (re.compile(r"\binc\.\b", re.IGNORECASE), "inc"),
    (re.compile(r"\bcorporation\b", re.IGNORECASE), "corp"),
    (re.compile(r"\bcorp\.\b", re.IGNORECASE), "corp"),
    (re.compile(r"\bcompany\b", re.IGNORECASE), "co"),
    (re.compile(r"\bco\.\b", re.IGNORECASE), "co"),
    (re.compile(r"\bl\.?l\.?c\.?\b", re.IGNORECASE), "llc"),
    (re.compile(r"\bl\.?l\.?p\.?\b", re.IGNORECASE), "llp"),
    (re.compile(r"\bp\.?l\.?l\.?c\.?\b", re.IGNORECASE), "pllc"),
    # French
    (re.compile(r"\bs\.?a\.?r\.?l\.?\b", re.IGNORECASE), "sarl"),
    (re.compile(r"\bs\.?a\.?s\.?\b", re.IGNORECASE), "sas"),
    (re.compile(r"\bs\.?a\.\b", re.IGNORECASE), "sa"),
    (re.compile(r"\bs\.?c\.?i\.?\b", re.IGNORECASE), "sci"),
    (re.compile(r"\be\.?u\.?r\.?l\.?\b", re.IGNORECASE), "eurl"),
    (re.compile(r"\bcie\b", re.IGNORECASE), "cie"),
]

# Common address thoroughfare & unit abbreviations to standardize
ADDRESS_ABBR_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bstreets?\b", re.IGNORECASE), "st"),
    (re.compile(r"\broads?\b", re.IGNORECASE), "rd"),
    (re.compile(r"\bavenues?\b", re.IGNORECASE), "ave"),
    (re.compile(r"\bdrives?\b", re.IGNORECASE), "dr"),
    (re.compile(r"\bboulevards?\b", re.IGNORECASE), "blvd"),
    (re.compile(r"\blanes?\b", re.IGNORECASE), "ln"),
    (re.compile(r"\bcourts?\b", re.IGNORECASE), "ct"),
    (re.compile(r"\bcircles?\b", re.IGNORECASE), "cir"),
    (re.compile(r"\bterraces?\b", re.IGNORECASE), "ter"),
    (re.compile(r"\bplaces?\b", re.IGNORECASE), "pl"),
    (re.compile(r"\bparkways?\b", re.IGNORECASE), "pkwy"),
    (re.compile(r"\bhighways?\b", re.IGNORECASE), "hwy"),
    (re.compile(r"\bsuites?\b", re.IGNORECASE), "ste"),
    (re.compile(r"\bapartments?\b", re.IGNORECASE), "apt"),
    (re.compile(r"\bfloors?\b", re.IGNORECASE), "fl"),
    (re.compile(r"\bbuildings?\b", re.IGNORECASE), "bldg"),
    (re.compile(r"\bp\.?o\.?\s*box\b", re.IGNORECASE), "po box"),
    (re.compile(r"\bpost\s+office\s+box\b", re.IGNORECASE), "po box"),
    # French thoroughfare types
    (re.compile(r"\b(bd|bvd)\.?\b", re.IGNORECASE), "blvd"),
    (re.compile(r"\br\.\s*", re.IGNORECASE), "rue "),
    (re.compile(r"\bav\.\s*", re.IGNORECASE), "ave "),
]

# Common stopwords and suffixes for token filtering
LEGAL_SUFFIXES: Set[str] = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "co", "company", "enterprises", "solutions", "group", "services", "tech",
    "technologies", "holdings", "partners", "assoc", "associates", "lp", "llp",
    "pllc", "pc", "pa", "foundation", "institute", "center", "centre", "club",
    "pvt", "private", "pte", "m/s", "ms",
    "sa", "sarl", "sas", "sasu", "snc", "sci", "sca", "eurl", "gie"
}

ADDRESS_STOPWORDS: Set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "parkway", "pkwy", "highway", "hwy",
    "way", "circle", "cir", "suite", "ste", "apartment", "apt", "floor", "fl",
    "building", "bldg", "unit", "block", "blk", "sector", "sec", "phase",
    "near", "opp", "opposite", "behind", "beside", "at", "post", "po", "box",
    "north", "south", "east", "west", "n", "s", "e", "w",
    "nagar", "colony", "marg", "chowk", "bazar", "bazaar", "gali", "mohalla",
    "rue", "avenue", "boulevard", "place", "route", "chemin", "allee",
    "the", "and", "of", "for", "in", "on", "de", "du", "des", "la", "le", "les", "et"
}


@dataclass(frozen=True)
class NormalizedEntityRecord:
    """
    Representation of an entity with both original raw columns and
    new normalized representations.
    """
    entity_id: str                      # Original raw ID
    business_name: str                  # Original raw name
    business_address: str               # Original raw address
    country: str                        # Original raw country
    normalized_name: str                # Standardized, lowercase normalized name
    normalized_address: str             # Standardized, lowercase normalized address
    clean_name: str                     # Compact alphanumeric-only representation
    address_numbers: Tuple[str, ...]    # Extracted numeric tokens from address


def strip_accents_latin(text: str) -> str:
    """
    Decompose Unicode accents from Latin-derived letters (e.g. é -> e, à -> a, Í -> I)
    while strictly preserving non-Latin native Indic scripts (Devanagari, Tamil, Telugu).
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    chars = []
    prev_is_latin = False
    for c in nfkd:
        if unicodedata.combining(c):
            # Only strip combining diacritics if modifying a Latin character
            if not prev_is_latin:
                chars.append(c)
        else:
            chars.append(c)
            prev_is_latin = ("a" <= c.lower() <= "z")
    return unicodedata.normalize("NFC", "".join(chars))


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (newlines, tabs, multiple spaces, NBSP) into single space."""
    if not text:
        return ""
    return RE_WHITESPACE.sub(" ", str(text)).strip()


def strip_noise_symbols(text: str) -> str:
    """Remove leading and trailing non-alphanumeric noise symbols (e.g. '--', '<<', '##', ',')."""
    if not text:
        return ""
    t = RE_LEADING_NOISE.sub("", str(text))
    t = RE_TRAILING_NOISE.sub("", t)
    return t.strip()


def normalize_punctuation(text: str) -> str:
    """Standardize common commercial punctuation symbols."""
    if not text:
        return ""
    t = str(text)
    t = t.replace("&", " and ")
    t = t.replace("@", " at ")
    t = t.replace("#", " unit ")
    # Replace curly quotes with standard quote
    t = t.replace("“", "'").replace("”", "'").replace("‘", "'").replace("’", "'")
    return t


def normalize_domain_name(name: str) -> str:
    """
    If a business name is formatted as a web domain (e.g. 'wilfordhancock.com'),
    extract and normalize the core business name.
    """
    if not name:
        return ""
    clean = name.strip()
    if RE_DOMAIN_SUFFIX.search(clean) or clean.lower().startswith("www."):
        clean = RE_WWW_PREFIX.sub("", clean)
        clean = RE_DOMAIN_SUFFIX.sub("", clean)
        clean = clean.replace(".", " ")
    return clean


def standardize_business_suffixes(name: str) -> str:
    """Standardize legal suffix variations to canonical forms based on training data."""
    if not name:
        return ""
    res = name
    for pattern, replacement in BUSINESS_SUFFIX_RULES:
        res = pattern.sub(replacement, res)
    return res


def standardize_address_abbreviations(addr: str) -> str:
    """Standardize street and unit abbreviations without removing numbers or locality info."""
    if not addr:
        return ""
    res = addr
    for pattern, replacement in ADDRESS_ABBR_RULES:
        res = pattern.sub(replacement, res)
    return res


def extract_address_numbers(addr: str) -> List[str]:
    """
    Extract numeric sequences from an address (e.g. house number, PIN code, suite).
    Does NOT modify or remove them from the address string.
    """
    if not addr:
        return []
    nums = RE_NUMBERS.findall(str(addr))
    return [n.lstrip("0") or "0" for n in nums if len(n) >= 1]


def clean_alphanumeric(text: str) -> str:
    """
    Extract contiguous lowercase alphanumeric characters (no spaces or punctuation).
    Used for compact exact matching, prefix trees, and URL alignment.
    """
    if not text:
        return ""
    text_latin = strip_accents_latin(str(text)).lower()
    return "".join(c for c in text_latin if c.isalnum())


def normalize_business_name(name: str, standardize_suffixes: bool = True) -> str:
    """
    Full normalization pipeline for business names:
    1. Smart Latin accent normalization (preserves Indic scripts).
    2. Lowercasing.
    3. Noise symbol stripping (leading '--', '<<', etc.).
    4. Commercial punctuation normalization (& -> and, @ -> at).
    5. Web domain normalization.
    6. Canonical business suffix standardization.
    7. Whitespace normalization.
    """
    if not name:
        return ""

    t = strip_accents_latin(str(name)).lower()
    t = strip_noise_symbols(t)
    t = normalize_punctuation(t)
    t = normalize_domain_name(t)

    if standardize_suffixes:
        t = standardize_business_suffixes(t)

    # Clean remaining special characters to spaces, preserving word characters and Indic script
    t = RE_NON_ALPHANUM_UNICODE.sub(" ", t)
    return normalize_whitespace(t)


def normalize_business_address(address: str, standardize_abbr: bool = True) -> str:
    """
    Full normalization pipeline for business addresses:
    1. Smart Latin accent normalization (preserves Indic scripts).
    2. Lowercasing.
    3. Noise symbol stripping.
    4. Punctuation normalization.
    5. Road & thoroughfare abbreviation standardization.
    6. Preserves all numbers and locality/city/state tokens.
    7. Whitespace normalization.
    """
    if not address:
        return ""

    t = strip_accents_latin(str(address)).lower()
    t = strip_noise_symbols(t)
    t = normalize_punctuation(t)

    if standardize_abbr:
        t = standardize_address_abbreviations(t)

    # Replace punctuation separators with spaces, preserving alphanumeric and Indic scripts
    t = RE_NON_ALPHANUM_UNICODE.sub(" ", t)
    return normalize_whitespace(t)


def normalize_record(record: Union[EntityRecord, Dict[str, str]]) -> NormalizedEntityRecord:
    """
    Normalize an entity record while preserving original raw values.
    Returns a NormalizedEntityRecord containing both original and normalized fields.
    """
    if isinstance(record, EntityRecord):
        eid = record.entity_id
        name = record.business_name
        addr = record.business_address
        country = record.country
    else:
        eid = record.get("entity_id", "")
        name = record.get("business_name", "")
        addr = record.get("business_address", "")
        country = record.get("country", "")

    norm_name = normalize_business_name(name)
    norm_addr = normalize_business_address(addr)
    c_name = clean_alphanumeric(name)
    nums = tuple(extract_address_numbers(addr))

    return NormalizedEntityRecord(
        entity_id=eid,
        business_name=name,
        business_address=addr,
        country=country,
        normalized_name=norm_name,
        normalized_address=norm_addr,
        clean_name=c_name,
        address_numbers=nums,
    )


def normalize_dataframe(df: "pandas.DataFrame") -> "pandas.DataFrame":
    """
    Take a DataFrame with columns [entity_id, business_name, business_address, country],
    preserve the original columns intact, and add 4 new normalized columns:
    - normalized_name
    - normalized_address
    - clean_name
    - address_numbers
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for normalize_dataframe.")

    result_df = df.copy()

    result_df["normalized_name"] = result_df["business_name"].apply(normalize_business_name)
    result_df["normalized_address"] = result_df["business_address"].apply(normalize_business_address)
    result_df["clean_name"] = result_df["business_name"].apply(clean_alphanumeric)
    result_df["address_numbers"] = result_df["business_address"].apply(
        lambda a: tuple(extract_address_numbers(a))
    )

    return result_df


# ---------------------------------------------------------------------------
# Backward Compatibility Helpers (used by features.py and blocking.py)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Alias for backwards compatibility with earlier feature extraction."""
    return normalize_business_name(text)


def extract_domain_core(text: str) -> str:
    """Extract clean domain core for inverted indexing."""
    clean = str(text).strip().lower()
    if RE_DOMAIN_SUFFIX.search(clean) or clean.startswith("www."):
        clean = RE_WWW_PREFIX.sub("", clean)
        clean = RE_DOMAIN_SUFFIX.sub("", clean)
        return clean_alphanumeric(clean)
    return ""


def extract_name_tokens(name: str) -> List[str]:
    """Extract informative name tokens excluding generic legal suffixes."""
    norm = normalize_business_name(name)
    tokens = norm.split()
    return [t for t in tokens if len(t) >= 2 and t not in LEGAL_SUFFIXES]


def extract_numbers(address: str) -> List[str]:
    """Alias for extract_address_numbers."""
    return extract_address_numbers(address)


def extract_address_tokens(address: str) -> List[str]:
    """Extract informative address tokens excluding stopwords and pure digits."""
    norm = normalize_business_address(address)
    tokens = norm.split()
    return [
        t for t in tokens
        if len(t) >= 3 and t not in ADDRESS_STOPWORDS and not t.isdigit()
    ]
