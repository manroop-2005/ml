"""
Data loading module for Business Entity Resolution Challenge.

Requirements addressed:
1. Load TSVs strictly with tab delimiters (preserving comma-containing addresses and IDs).
2. Preserve original raw values (no unwanted conversions, stripping, or NaN coercion).
3. Validate required columns for source files and ground truth files.
4. Validate expected S1/S2/S3 entity ID prefixes.
5. Do not silently drop rows (raise clear errors on malformed lines).
6. Provide actionable error messages indicating file, line number, and offending content.
7. Fully configurable paths and streaming/batch loading support.
"""

import os
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

# Expected Headers
SOURCE_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)

GROUND_TRUTH_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "source1_entity_id",
    "matched_entity_ids",
)

VALID_SOURCE_PREFIXES: Tuple[str, ...] = ("S1-", "S2-", "S3-")
VALID_MATCH_PREFIXES: Tuple[str, ...] = ("S2-", "S3-")
DELIMITER: str = "\t"


# Custom Exceptions
class DataLoaderError(Exception):
    """Base exception for data loading and validation errors."""
    pass


class FileFormatError(DataLoaderError):
    """Raised when file structure, delimiter, or header is broken."""
    pass


class MissingColumnError(DataLoaderError):
    """Raised when required columns are absent or mismatched."""
    pass


class MalformedRowError(DataLoaderError):
    """Raised when a specific data row has incorrect column count or delimiter issues."""
    def __init__(self, file_path: str, line_number: int, line_content: str, reason: str):
        self.file_path = file_path
        self.line_number = line_number
        self.line_content = line_content
        self.reason = reason
        super().__init__(
            f"Malformed row in '{file_path}' at line {line_number}: {reason}\n"
            f"Offending line content: {line_content!r}"
        )


class InvalidEntityIdError(DataLoaderError):
    """Raised when an entity ID violates prefix or formatting constraints."""
    def __init__(self, file_path: str, line_number: int, entity_id: str, expected_prefix: Optional[str] = None):
        self.file_path = file_path
        self.line_number = line_number
        self.entity_id = entity_id
        self.expected_prefix = expected_prefix
        msg = f"Invalid entity ID {entity_id!r} in '{file_path}' at line {line_number}."
        if expected_prefix:
            msg += f" Expected prefix: {expected_prefix!r}."
        else:
            msg += f" Expected one of valid prefixes: {VALID_SOURCE_PREFIXES}."
        super().__init__(msg)


class GroundTruthFormatError(DataLoaderError):
    """Raised when ground truth records contain self-matches, invalid prefixes, or duplicates."""
    pass


@dataclass(frozen=True)
class EntityRecord:
    """Immutable representation of a raw entity record."""
    entity_id: str
    business_name: str
    business_address: str
    country: str


@dataclass(frozen=True)
class GroundTruthRecord:
    """Immutable representation of a ground truth match record."""
    source1_entity_id: str
    matched_entity_ids: Tuple[str, ...]


def infer_expected_prefix(file_path: Union[str, Path]) -> Optional[str]:
    """Infer expected entity prefix (S1-, S2-, S3-) from filename."""
    name = Path(file_path).name.lower()
    if "source1" in name:
        return "S1-"
    elif "source2" in name:
        return "S2-"
    elif "source3" in name:
        return "S3-"
    return None


def validate_header(
    header_line: str,
    file_path: str,
    expected_columns: Sequence[str]
) -> List[str]:
    """
    Validate TSV header row.
    Detects comma-separated mistakes, missing tabs, and missing required columns.
    """
    if not header_line:
        raise FileFormatError(f"File '{file_path}' is empty.")

    # Catch common mistake: comma-separated file instead of tab-separated
    if DELIMITER not in header_line and "," in header_line:
        raise FileFormatError(
            f"File '{file_path}' header contains commas but no TAB (\\t) delimiters. "
            "All challenge files must be strictly tab-separated (.tsv)."
        )

    columns = [col.strip() for col in header_line.rstrip("\r\n").split(DELIMITER)]

    if columns != list(expected_columns):
        raise MissingColumnError(
            f"File '{file_path}' has invalid header columns: {columns}. "
            f"Expected exactly: {list(expected_columns)} (tab-separated)."
        )

    return columns


def stream_source_file(
    file_path: Union[str, Path],
    expected_prefix: Optional[str] = "infer",
    limit: Optional[int] = None,
    encoding: str = "utf-8"
) -> Generator[EntityRecord, None, None]:
    """
    Stream entity records row-by-row from a source TSV file.
    Does not load the entire file into memory, ideal for large files.
    Preserves exact raw values.
    """
    file_str = str(file_path)
    if not os.path.isfile(file_str):
        raise FileNotFoundError(f"Source file not found at: '{file_str}'")

    if expected_prefix == "infer":
        prefix_to_check = infer_expected_prefix(file_str)
    else:
        prefix_to_check = expected_prefix

    expected_col_count = len(SOURCE_REQUIRED_COLUMNS)

    with open(file_str, mode="r", encoding=encoding, newline="") as f:
        # Read and validate header
        first_line = f.readline()
        validate_header(first_line, file_str, SOURCE_REQUIRED_COLUMNS)

        count = 0
        for line_num, raw_line in enumerate(f, start=2):
            if not raw_line.strip():
                # Skip blank empty lines at end of file if any
                continue

            # Strip trailing newline only, preserve raw spaces within fields
            line = raw_line.rstrip("\r\n")

            # Parse with explicit tab delimiter
            parts = line.split(DELIMITER)
            if len(parts) != expected_col_count:
                raise MalformedRowError(
                    file_path=file_str,
                    line_number=line_num,
                    line_content=line,
                    reason=f"Expected {expected_col_count} columns, found {len(parts)}"
                )

            entity_id, business_name, business_address, country = parts

            # Validate entity_id
            if not entity_id:
                raise MalformedRowError(
                    file_path=file_str,
                    line_number=line_num,
                    line_content=line,
                    reason="Field 'entity_id' is empty"
                )

            if prefix_to_check:
                if not entity_id.startswith(prefix_to_check):
                    raise InvalidEntityIdError(
                        file_path=file_str,
                        line_number=line_num,
                        entity_id=entity_id,
                        expected_prefix=prefix_to_check
                    )
            else:
                if not any(entity_id.startswith(p) for p in VALID_SOURCE_PREFIXES):
                    raise InvalidEntityIdError(
                        file_path=file_str,
                        line_number=line_num,
                        entity_id=entity_id,
                        expected_prefix=None
                    )

            # Preserve raw values exactly
            yield EntityRecord(
                entity_id=entity_id,
                business_name=business_name,
                business_address=business_address,
                country=country,
            )

            count += 1
            if limit is not None and count >= limit:
                break


def load_source_file(
    file_path: Union[str, Path],
    expected_prefix: Optional[str] = "infer",
    limit: Optional[int] = None,
    as_dataframe: bool = False,
    encoding: str = "utf-8"
) -> Union[List[EntityRecord], "pandas.DataFrame"]:
    """
    Load records from a source TSV file into memory as a list of EntityRecords
    or as a pandas DataFrame preserving raw string values.
    """
    records = list(
        stream_source_file(
            file_path=file_path,
            expected_prefix=expected_prefix,
            limit=limit,
            encoding=encoding
        )
    )

    if as_dataframe:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load records as a DataFrame. "
                "Install pandas using 'pip install pandas' or set as_dataframe=False."
            )
        data = [
            {
                "entity_id": r.entity_id,
                "business_name": r.business_name,
                "business_address": r.business_address,
                "country": r.country,
            }
            for r in records
        ]
        # Ensure strings are preserved without NaN conversion
        df = pd.DataFrame(data, columns=list(SOURCE_REQUIRED_COLUMNS), dtype=str)
        return df

    return records


def stream_ground_truth(
    file_path: Union[str, Path],
    limit: Optional[int] = None,
    encoding: str = "utf-8"
) -> Generator[GroundTruthRecord, None, None]:
    """
    Stream ground truth records row-by-row.
    Validates S1 prefix, S2/S3 match prefixes, self-matches, and intra-list duplicates.
    """
    file_str = str(file_path)
    if not os.path.isfile(file_str):
        raise FileNotFoundError(f"Ground truth file not found at: '{file_str}'")

    expected_col_count = len(GROUND_TRUTH_REQUIRED_COLUMNS)

    with open(file_str, mode="r", encoding=encoding, newline="") as f:
        first_line = f.readline()
        validate_header(first_line, file_str, GROUND_TRUTH_REQUIRED_COLUMNS)

        count = 0
        seen_s1_ids: Set[str] = set()

        for line_num, raw_line in enumerate(f, start=2):
            if not raw_line.strip():
                continue

            line = raw_line.rstrip("\r\n")
            parts = line.split(DELIMITER)
            if len(parts) != expected_col_count:
                raise MalformedRowError(
                    file_path=file_str,
                    line_number=line_num,
                    line_content=line,
                    reason=f"Expected {expected_col_count} columns, found {len(parts)}"
                )

            s1_id, matched_str = parts

            if not s1_id.startswith("S1-"):
                raise InvalidEntityIdError(
                    file_path=file_str,
                    line_number=line_num,
                    entity_id=s1_id,
                    expected_prefix="S1-"
                )

            if s1_id in seen_s1_ids:
                raise GroundTruthFormatError(
                    f"Duplicate source1_entity_id {s1_id!r} found at line {line_num} in '{file_str}'."
                )
            seen_s1_ids.add(s1_id)

            if matched_str:
                matched_ids = matched_str.split(",")
                # Check for duplicates within list
                if len(matched_ids) != len(set(matched_ids)):
                    raise GroundTruthFormatError(
                        f"Repeated ID within match list for {s1_id!r} at line {line_num} in '{file_str}': {matched_str!r}"
                    )

                for mid in matched_ids:
                    if mid.startswith("S1-"):
                        raise GroundTruthFormatError(
                            f"Self-match error at line {line_num} in '{file_str}': "
                            f"{s1_id} matches Source-1 ID {mid!r}. Only S2- and S3- allowed."
                        )
                    if not (mid.startswith("S2-") or mid.startswith("S3-")):
                        raise GroundTruthFormatError(
                            f"Invalid match ID prefix at line {line_num} in '{file_str}': {mid!r}. "
                            "Expected prefix 'S2-' or 'S3-'."
                        )
                matches_tuple = tuple(matched_ids)
            else:
                matches_tuple = ()

            yield GroundTruthRecord(
                source1_entity_id=s1_id,
                matched_entity_ids=matches_tuple
            )

            count += 1
            if limit is not None and count >= limit:
                break


def load_ground_truth(
    file_path: Union[str, Path],
    limit: Optional[int] = None,
    as_dict: bool = True,
    encoding: str = "utf-8"
) -> Union[Dict[str, Set[str]], List[GroundTruthRecord]]:
    """
    Load ground truth records.
    If as_dict=True (default), returns {source1_entity_id: set(matched_ids)}.
    Otherwise returns list of GroundTruthRecord dataclasses.
    """
    records = list(stream_ground_truth(file_path=file_path, limit=limit, encoding=encoding))
    if as_dict:
        return {r.source1_entity_id: set(r.matched_entity_ids) for r in records}
    return records


def load_dataset_split(
    split_dir: Union[str, Path],
    is_train: bool = True,
    limit_per_file: Optional[int] = None
) -> Dict[str, Union[List[EntityRecord], Dict[str, Set[str]]]]:
    """
    Load all source files (and ground truth if is_train=True) from a directory.
    Configurable for any dataset folder.
    """
    split_path = Path(split_dir)
    prefix = "train" if is_train else "test"

    s1_path = split_path / f"{prefix}_source1.tsv"
    s2_path = split_path / f"{prefix}_source2.tsv"
    s3_path = split_path / f"{prefix}_source3.tsv"

    data = {
        "source1": load_source_file(s1_path, expected_prefix="S1-", limit=limit_per_file),
        "source2": load_source_file(s2_path, expected_prefix="S2-", limit=limit_per_file),
        "source3": load_source_file(s3_path, expected_prefix="S3-", limit=limit_per_file),
    }

    if is_train:
        gt_path = split_path / "train_ground_truth.tsv"
        data["ground_truth"] = load_ground_truth(gt_path, limit=limit_per_file)

    return data
