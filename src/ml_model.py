
"""
Supply Chain Risk Intelligence Platform
Milestone 3: Machine Learning

Predicts the Late_delivery_risk label using historical
order information.

Models:
1. Logistic Regression
2. Random Forest

Leakage prevention:
- Excludes actual shipping duration.
- Excludes delivery status and post-delivery information.
- Excludes customer identifiers and personal information.
"""

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "supply_chain_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Late_delivery_risk"

RANDOM_STATE = 42
SAMPLE_SIZE = 50_000


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the cleaned dataset and validate the target."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run src/data_ingestion.py first."
        )

    print("Loading cleaned supply-chain dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[TARGET]).copy()

    if df.empty:
        raise ValueError("No records have a valid target label.")

    # Remove labels that cannot be interpreted as numbers.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    labels = set(df[TARGET].unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary target values 0 and 1; found {labels}"
        )

    df[TARGET] = df[TARGET].astype(int)

    print(f"Available records: {len(df):,}")
    print(f"Target distribution:\n{df[TARGET].value_counts().sort_index()}")

    return df


# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

def select_features(df: pd.DataFrame):
    """
    Select features available before the delivery outcome.

    Deliberately exclude:
    - Days for shipping (real)
    - Delivery Status
    - Shipping date
    - Customer identifiers and personal details
    - Order identifiers and product identifiers
    """

    numeric_candidates = [
        "Days for shipment (scheduled)",
        "Sales",
        "Order Item Quantity",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Profit Ratio",
    ]

    categorical_candidates = [
        "Type",
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Country",
        "Customer Segment",
        "Department Name",
        "Category Name",
    ]

    numeric_columns = [
        column
        for column in numeric_candidates
        if column in df.columns
    ]

    categorical_columns = [
        column
        for column in categorical_candidates
        if column in df.columns
    ]

    feature_columns = numeric_columns + categorical_columns

    if not numeric_columns or not categorical_columns:
        print(
            "Note: Some expected columns are missing. "
            "Using the available candidate columns."
        )

    if not feature_columns:
        raise ValueError(
            "No usable candidate features were found. "
            "Inspect the dataset column names."
        )

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    # Convert numeric fields safely.
    for column in numeric_columns:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

    # Ensure categorical columns are handled consistently.
    for column in categorical_columns:
        X[column] = X[column].fillna("Missing").astype(str)

    print("\nSelected features:")
    for column in feature_columns:
        print(f"  - {column}")

    print(f"\nNumeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    return X, y, numeric_columns, categorical_columns


# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

def build_preprocessor(numeric_columns, categorical_columns):
    """Build preprocessing pipelines for both feature types."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


# --------------------------------------------------
# 5. MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(name, pipeline, X_test, y_test):
    """Calculate metrics and print the classification report."""

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return metrics, predictions


# --------------------------------------------------
# 6. MAIN TRAINING PIPELINE
# --------------------------------------------------

def main() -> int:
    try:
        df = load_data()

        X, y, numeric_columns, categorical_columns = (
            select_features(df)
        )

        # Use a reproducible sample to keep initial training
        # manageable on a typical personal computer.
        if len(X) > SAMPLE_SIZE:
            sampled_indices = X.sample(
                n=SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            ).index

            X = X.loc[sampled_indices].copy()
            y = y.loc[sampled_indices].copy()

            print(
                f"\nUsing a reproducible sample of "
                f"{SAMPLE_SIZE:,} records."
            )

        if y.nunique() != 2 or y.value_counts().min() < 2:
            raise ValueError(
                "Not enough examples of both classes for "
                "stratified training and testing."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print(f"\nTraining records: {len(X_train):,}")
        print(f"Testing records: {len(X_test):,}")

        results = []
        fitted_models = {}

        model_definitions = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        }

        for name, estimator in model_definitions.items():
            print(f"\nTraining {name}...")

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        build_preprocessor(
                            numeric_columns,
                            categorical_columns,
                        ),
                    ),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)

            metrics, predictions = evaluate_model(
                name,
                pipeline,
                X_test,
                y_test,
            )

            results.append(metrics)
            fitted_models[name] = pipeline

            # Save each trained pipeline, including preprocessing.
            safe_name = name.lower().replace(" ", "_")

            model_path = (
                MODEL_DIR / f"{safe_name}_delivery_risk.joblib"
            )

            joblib.dump(pipeline, model_path)
            print(f"Saved model: {model_path}")

            # Save a confusion matrix for each model.
            cm = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            )

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[0, 1],
            )

            display.plot(values_format="d")
            plt.title(f"{name} - Confusion Matrix")
            plt.tight_layout()

            figure_path = (
                FIGURE_DIR
                / f"{safe_name}_confusion_matrix.png"
            )

            plt.savefig(figure_path, dpi=150)
            plt.close()

            print(f"Saved confusion matrix: {figure_path}")

        # Compare models.
        results_df = pd.DataFrame(results).sort_values(
            "F1 Score",
            ascending=False,
        )

        metrics_path = REPORT_DIR / "ml_model_comparison.csv"
        results_df.to_csv(metrics_path, index=False)

        report_path = REPORT_DIR / "ml_model_report.txt"

        report_lines = [
            "SUPPLY CHAIN RISK INTELLIGENCE PLATFORM",
            "MACHINE LEARNING MODEL COMPARISON",
            "=" * 65,
            "",
            f"Dataset: {DATA_PATH.name}",
            f"Sample size: {len(X):,}",
            f"Training records: {len(X_train):,}",
            f"Testing records: {len(X_test):,}",
            f"Target: {TARGET}",
            "",
            "FEATURES USED",
            "-" * 65,
            *[f"- {column}" for column in X.columns],
            "",
            "MODEL METRICS",
            "-" * 65,
            results_df.to_string(index=False),
            "",
            "INTERPRETATION NOTES",
            "-" * 65,
            "- Accuracy is the overall fraction classified correctly.",
            "- Precision measures how many predicted positives were correct.",
            "- Recall measures how many actual positive cases were found.",
            "- F1 balances precision and recall.",
            "- ROC-AUC measures ranking discrimination across thresholds.",
            "- These metrics describe a held-out test set, not guaranteed future performance.",
            "- The dataset label may encode historical rules; validate its definition before production use.",
        ]

        report_path.write_text(
            "\n".join(report_lines),
            encoding="utf-8",
        )

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(results_df.to_string(index=False))

        print(f"\nComparison CSV: {metrics_path}")
        print(f"Text report: {report_path}")
        print(f"Saved models: {MODEL_DIR}")
        print("\nMachine learning training completed successfully!")

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        KeyError,
    ) as error:
        print(f"\nMACHINE LEARNING FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
