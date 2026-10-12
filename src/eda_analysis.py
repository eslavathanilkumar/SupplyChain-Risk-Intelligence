
"""
Supply Chain Risk Intelligence Platform
Module: Exploratory Data Analysis (EDA)

Loads the cleaned DataCo dataset and generates:
- Dataset overview and statistical summaries
- Missing-value and late-delivery analysis
- Shipping-mode and regional analysis
- Sales and shipping-delay visualizations
- A text-based EDA report
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "supply_chain_cleaned.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_PATH = REPORT_DIR / "eda_report.txt"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """Load the cleaned dataset."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{file_path}\n\n"
            "Run src/data_ingestion.py successfully first."
        )

    print("Loading cleaned dataset...")
    df = pd.read_csv(file_path, low_memory=False)

    # Normalize whitespace in column names.
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    return df


# --------------------------------------------------
# 3. COLUMN AND DATA PREPARATION
# --------------------------------------------------

def find_column(df: pd.DataFrame, *candidates: str):
    """Find a column using case-insensitive matching."""

    normalized = {
        str(column).strip().casefold(): column
        for column in df.columns
    }

    for candidate in candidates:
        found = normalized.get(candidate.strip().casefold())
        if found is not None:
            return found

    return None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert selected numeric columns when available."""

    numeric_candidates = [
        "Sales",
        "Order Item Quantity",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Late_delivery_risk",
        "Benefit per order",
        "Order Item Product Price",
        "Order Item Discount",
        "Order Item Total",
        "Product Price",
        "Customer Id",
        "Order Id",
    ]

    for candidate in numeric_candidates:
        column = find_column(df, candidate)

        if column is not None:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    return df


# --------------------------------------------------
# 4. CHART UTILITIES
# --------------------------------------------------

def save_chart(filename: str) -> None:
    """Save the current chart and release its resources."""

    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Chart saved: {output_path.name}")


def plot_categorical_counts(
    df: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
    top_n: int = 10,
) -> None:
    """Plot the most frequent categories."""

    counts = df[column].fillna("Missing").astype(str).value_counts()
    counts = counts.head(top_n)

    if counts.empty:
        return

    plt.figure(figsize=(11, 6))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(title)
    plt.xlabel("Number of records")
    plt.ylabel(column)

    save_chart(filename)


# --------------------------------------------------
# 5. GENERATE ANALYSIS AND CHARTS
# --------------------------------------------------

def run_eda(df: pd.DataFrame) -> str:
    """Generate summaries, visualizations and a text report."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "SUPPLY CHAIN RISK INTELLIGENCE PLATFORM",
        "EXPLORATORY DATA ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Total rows: {len(df):,}",
        f"Total columns: {len(df.columns):,}",
        f"Exact duplicate rows: {int(df.duplicated().sum()):,}",
        f"Total missing cells: {int(df.isna().sum().sum()):,}",
        "",
        "COLUMN NAMES",
        "-" * 70,
        ", ".join(str(column) for column in df.columns),
        "",
        "DATA TYPES",
        "-" * 70,
        df.dtypes.to_string(),
        "",
        "NUMERIC SUMMARY",
        "-" * 70,
        df.describe(include="number").to_string(),
        "",
        "MISSING VALUES BY COLUMN",
        "-" * 70,
        df.isna().sum().sort_values(ascending=False).head(30).to_string(),
    ]

    # Chart 1: Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(20)

    if not missing.empty:
        plt.figure(figsize=(12, 7))
        sns.barplot(x=missing.values, y=missing.index)
        plt.title("Top 20 Columns by Missing Values")
        plt.xlabel("Missing cells")
        plt.ylabel("Column")
        save_chart("missing_values.png")
    else:
        lines.append("\nNo missing values found.")

    # Chart 2: Late-delivery risk
    risk_column = find_column(df, "Late_delivery_risk")

    if risk_column is not None:
        risk_counts = (
            df[risk_column]
            .fillna(-1)
            .astype(str)
            .value_counts()
        )

        plt.figure(figsize=(8, 5))
        sns.barplot(x=risk_counts.index, y=risk_counts.values)
        plt.title("Late-Delivery Risk Distribution")
        plt.xlabel("Late-delivery risk label")
        plt.ylabel("Number of orders")
        save_chart("late_delivery_risk.png")

        lines.extend([
            "",
            "LATE-DELIVERY RISK DISTRIBUTION",
            "-" * 70,
            df[risk_column].value_counts(dropna=False).to_string(),
        ])
    else:
        lines.append(
            "\nLate_delivery_risk column not found; "
            "the risk chart was skipped."
        )

    # Chart 3: Shipping modes
    shipping_column = find_column(df, "Shipping Mode")

    if shipping_column is not None:
        plot_categorical_counts(
            df,
            shipping_column,
            "shipping_mode_distribution.png",
            "Orders by Shipping Mode",
        )

        lines.extend([
            "",
            "ORDERS BY SHIPPING MODE",
            "-" * 70,
            df[shipping_column].value_counts(dropna=False).to_string(),
        ])

    # Chart 4: Order regions
    region_column = find_column(df, "Order Region")

    if region_column is not None:
        plot_categorical_counts(
            df,
            region_column,
            "order_region_distribution.png",
            "Orders by Region (Top 10)",
            top_n=10,
        )

        lines.extend([
            "",
            "TOP 10 ORDER REGIONS",
            "-" * 70,
            df[region_column].value_counts().head(10).to_string(),
        ])

    # Chart 5: Actual versus scheduled shipping time
    actual_column = find_column(df, "Days for shipping (real)")
    scheduled_column = find_column(
        df, "Days for shipment (scheduled)"
    )

    if actual_column is not None and scheduled_column is not None:
        delay_data = df[[actual_column, scheduled_column]].dropna()

        if not delay_data.empty:
            delay_data = delay_data.rename(
                columns={
                    actual_column: "Actual shipping days",
                    scheduled_column: "Scheduled shipping days",
                }
            )

            delay_data["Delay days"] = (
                delay_data["Actual shipping days"]
                - delay_data["Scheduled shipping days"]
            )

            lines.extend([
                "",
                "SHIPPING DELAY ANALYSIS",
                "-" * 70,
                f"Orders with shipping-time data: {len(delay_data):,}",
                f"Late shipments: {(delay_data['Delay days'] > 0).sum():,}",
                f"On-time shipments: {(delay_data['Delay days'] == 0).sum():,}",
                f"Early shipments: {(delay_data['Delay days'] < 0).sum():,}",
                f"Average delay in days: {delay_data['Delay days'].mean():.3f}",
                f"Median delay in days: {delay_data['Delay days'].median():.3f}",
            ])

            # Keep the visualization readable for very large datasets.
            plot_data = delay_data.sample(
                n=min(10000, len(delay_data)),
                random_state=42,
            )

            plt.figure(figsize=(9, 6))
            sns.scatterplot(
                data=plot_data,
                x="Scheduled shipping days",
                y="Actual shipping days",
                alpha=0.25,
            )

            maximum = max(
                plot_data["Scheduled shipping days"].max(),
                plot_data["Actual shipping days"].max(),
            )

            plt.plot(
                [0, maximum],
                [0, maximum],
                linestyle="--",
                label="Actual = scheduled",
            )

            plt.title("Actual vs Scheduled Shipping Time")
            plt.legend()
            save_chart("actual_vs_scheduled_shipping.png")

            plt.figure(figsize=(10, 6))
            sns.histplot(
                delay_data["Delay days"],
                bins=30,
                kde=False,
            )
            plt.title("Distribution of Shipping Delay (Days)")
            plt.xlabel("Actual days minus scheduled days")
            plt.ylabel("Number of orders")
            save_chart("shipping_delay_distribution.png")

    # Chart 6: Sales distribution
    sales_column = find_column(df, "Sales")

    if sales_column is not None:
        sales = df[sales_column].dropna()

        if not sales.empty:
            lines.extend([
                "",
                "SALES SUMMARY",
                "-" * 70,
                sales.describe().to_string(),
            ])

            plt.figure(figsize=(10, 6))
            sns.histplot(sales, bins=40, kde=False)
            plt.title("Sales Distribution")
            plt.xlabel("Sales")
            plt.ylabel("Number of records")
            save_chart("sales_distribution.png")

    # Correlation heatmap
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] >= 2:
        # Use a limited number of columns for readability.
        selected_columns = numeric_df.columns[:15]
        correlation = numeric_df[selected_columns].corr()

        plt.figure(figsize=(13, 10))
        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            annot=False,
        )
        plt.title("Numeric Feature Correlation Heatmap")
        save_chart("correlation_heatmap.png")

        lines.extend([
            "",
            "NUMERIC FEATURE CORRELATIONS",
            "-" * 70,
            correlation.round(3).to_string(),
        ])

    lines.extend([
        "",
        "NOTES",
        "-" * 70,
        "- These results describe historical records, not guaranteed future outcomes.",
        "- Correlation does not establish causation.",
        "- Missing values have not been automatically imputed.",
        "- Late-delivery risk is a dataset label; verify its meaning before modeling.",
    ])

    return "\n".join(lines)


# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("SUPPLY CHAIN EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    try:
        df = load_data(INPUT_PATH)
        df = prepare_data(df)

        report = run_eda(df)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")

        print(f"\nEDA report saved:\n{REPORT_PATH}")
        print(f"Charts saved in:\n{FIGURE_DIR}")
        print("\nEDA completed successfully!")

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"\nEDA FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
