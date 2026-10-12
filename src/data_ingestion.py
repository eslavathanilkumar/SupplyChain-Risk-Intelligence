
"""
Supply Chain Risk Intelligence Platform
Module: Data Ingestion and Validation

Responsibilities:
- Load the DataCo supply-chain dataset.
- Validate the dataset.
- Generate a data quality report.
- Remove exact duplicate records.
- Save the cleaned dataset.
"""

from pathlib import Path
import sys

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SOURCE_DIR = RAW_DIR / "DataCoSupplyChainDataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_PATH = SOURCE_DIR / "DataCoSupplyChainDataset.csv"
DESCRIPTION_PATH = SOURCE_DIR / "DescriptionDataCoSupplyChain.csv"

OUTPUT_PATH = PROCESSED_DIR / "supply_chain_cleaned.csv"
REPORT_PATH = PROCESSED_DIR / "data_quality_report.txt"


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the source CSV with encoding fallback."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}\n\n"
            "Check that the CSV is inside "
            "data/raw/DataCoSupplyChainDataset/."
        )

    print(f"\nLoading dataset:\n{file_path}")
    print("Please wait; the CSV is approximately 96 MB.")

    last_error = None

    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False,
            )
            print(f"Successfully loaded using {encoding} encoding.")
            return df

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f"Could not decode the dataset: {last_error}"
    )


# --------------------------------------------------
# 3. VALIDATE DATASET
# --------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the basic structure of the dataset."""

    if df.empty:
        raise ValueError("The dataset contains no records.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    # Remove accidental spaces from column names.
    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        duplicate_columns = df.columns[
            df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names detected: {duplicate_columns}"
        )

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


# --------------------------------------------------
# 4. DATA QUALITY REPORT
# --------------------------------------------------

def create_quality_report(df: pd.DataFrame) -> str:
    """Create a report of missing values, duplicates and types."""

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[
        missing_counts > 0
    ].sort_values(ascending=False)

    duplicate_count = int(df.duplicated().sum())
    total_missing = int(df.isna().sum().sum())

    report_lines = [
        "SUPPLY CHAIN RISK INTELLIGENCE PLATFORM",
        "DATA QUALITY REPORT",
        "=" * 65,
        "",
        f"Dataset file: {DATASET_PATH.name}",
        f"Total rows: {len(df):,}",
        f"Total columns: {len(df.columns):,}",
        f"Duplicate rows before cleaning: {duplicate_count:,}",
        f"Total missing cells: {total_missing:,}",
        "",
        "MISSING VALUES BY COLUMN",
        "-" * 65,
        (
            missing_counts.to_string()
            if not missing_counts.empty
            else "No missing values found."
        ),
        "",
        "COLUMN DATA TYPES",
        "-" * 65,
        df.dtypes.to_string(),
        "",
        "COLUMN NAMES",
        "-" * 65,
        "\n".join(str(column) for column in df.columns),
        "",
        "NUMERIC SUMMARY",
        "-" * 65,
        df.describe(include="number").to_string(),
    ]

    return "\n".join(report_lines)


# --------------------------------------------------
# 5. CLEAN DATASET
# --------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply conservative cleaning.

    Removes exact duplicate rows only.
    Missing values are retained and reported rather
    than automatically deleting useful records.
    """

    cleaned_df = df.copy()

    rows_before = len(cleaned_df)

    cleaned_df = cleaned_df.drop_duplicates()

    duplicates_removed = rows_before - len(cleaned_df)

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    print(f"Rows before cleaning: {rows_before:,}")
    print(f"Duplicate rows removed: {duplicates_removed:,}")
    print(f"Rows after cleaning: {len(cleaned_df):,}")

    missing_cells = int(cleaned_df.isna().sum().sum())
    print(f"Missing cells remaining: {missing_cells:,}")

    return cleaned_df


# --------------------------------------------------
# 6. SAVE OUTPUTS
# --------------------------------------------------

def save_outputs(
    cleaned_df: pd.DataFrame,
    quality_report: str,
) -> None:
    """Save the cleaned CSV and quality report."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    REPORT_PATH.write_text(
        quality_report,
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("OUTPUT FILES SAVED")
    print("=" * 60)
    print(f"Cleaned dataset:\n{OUTPUT_PATH}")
    print(f"\nQuality report:\n{REPORT_PATH}")


# --------------------------------------------------
# 7. MAIN PIPELINE
# --------------------------------------------------

def main() -> int:
    """Execute the complete data-ingestion pipeline."""

    print("=" * 60)
    print("SUPPLY CHAIN RISK INTELLIGENCE PLATFORM")
    print("DATA INGESTION PIPELINE")
    print("=" * 60)

    try:
        df = load_dataset(DATASET_PATH)

        validate_dataset(df)

        quality_report = create_quality_report(df)

        cleaned_df = clean_dataset(df)

        save_outputs(cleaned_df, quality_report)

        print("\nPreview of cleaned data:")
        print(cleaned_df.head(5).to_string(index=False))

        print("\nData ingestion completed successfully!")

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"\nPIPELINE FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())