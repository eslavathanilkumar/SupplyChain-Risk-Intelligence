
from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

ML_FILE = REPORT_DIR / "ml_model_comparison.csv"
NETWORK_FILE = REPORT_DIR / "country_network_metrics.csv"

OUTPUT_TEXT = REPORT_DIR / "risk_intelligence_report.txt"
OUTPUT_JSON = REPORT_DIR / "risk_intelligence_report.json"


def load_inputs():
    for path in [ML_FILE, NETWORK_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    ml = pd.read_csv(ML_FILE)
    network = pd.read_csv(NETWORK_FILE)

    required_ml = {
        "Model", "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC-AUC"
    }
    required_network = {
        "country", "in_degree", "out_degree",
        "in_strength_trade_value_usd",
        "out_strength_trade_value_usd",
        "betweenness_centrality"
    }

    if not required_ml.issubset(ml.columns):
        raise ValueError(
            "The ML comparison report has unexpected columns."
        )

    if not required_network.issubset(network.columns):
        raise ValueError(
            "The network metrics file has unexpected columns."
        )

    return ml, network


def build_insights(ml, network):
    insights = []

    # 1. Compare ML models using held-out test metrics.
    best_model = ml.sort_values(
        "ROC-AUC", ascending=False
    ).iloc[0]

    insights.append({
        "category": "delivery_risk_model",
        "severity": "information",
        "title": "Best model by ROC-AUC",
        "finding": (
            f"{best_model['Model']} achieved ROC-AUC "
            f"{best_model['ROC-AUC']:.3f}, F1 score "
            f"{best_model['F1 Score']:.3f}, and recall "
            f"{best_model['Recall']:.1%} on the evaluation set."
        ),
        "recommendation": (
            "Compare recall and precision at different "
            "classification thresholds before using predictions "
            "to prioritize operational interventions."
        ),
        "limitation": (
            "These metrics describe the available historical "
            "dataset and do not guarantee future performance."
        ),
    })

    # 2. Find the highest structural betweenness.
    top_betweenness = network.sort_values(
        "betweenness_centrality", ascending=False
    ).head(5)

    names = ", ".join(
        top_betweenness["country"].astype(str).tolist()
    )

    insights.append({
        "category": "network_centrality",
        "severity": "investigate",
        "title": "Countries with high betweenness centrality",
        "finding": (
            "The top five countries by betweenness centrality "
            f"in the current graph are: {names}."
        ),
        "recommendation": (
            "Inspect the trade connections around these countries "
            "and test how network connectivity changes when each "
            "node is removed."
        ),
        "limitation": (
            "Betweenness measures structural position, not proven "
            "economic impact or disruption probability."
        ),
    })

    # 3. Find countries with the largest recorded outgoing flows.
    top_exports = network.sort_values(
        "out_strength_trade_value_usd", ascending=False
    ).head(5)

    export_names = ", ".join(
        top_exports["country"].astype(str).tolist()
    )

    insights.append({
        "category": "trade_concentration",
        "severity": "investigate",
        "title": "Largest outgoing reported trade values",
        "finding": (
            "Countries with the largest summed outgoing trade "
            f"values in the current graph include: {export_names}."
        ),
        "recommendation": (
            "Review the underlying edge list to identify important "
            "trade partners and determine whether particular "
            "products or regions have concentrated dependencies."
        ),
        "limitation": (
            "The current graph may be incomplete because some API "
            "requests failed. Country-reported exports can also "
            "differ from partner-reported imports."
        ),
    })

    # 4. Find countries with the largest recorded incoming flows.
    top_imports = network.sort_values(
        "in_strength_trade_value_usd", ascending=False
    ).head(5)

    import_names = ", ".join(
        top_imports["country"].astype(str).tolist()
    )

    insights.append({
        "category": "trade_dependency",
        "severity": "investigate",
        "title": "Largest incoming reported trade values",
        "finding": (
            "Countries with the largest summed incoming trade "
            f"values in the current graph include: {import_names}."
        ),
        "recommendation": (
            "Investigate the partners and product categories "
            "behind these flows before drawing conclusions about "
            "dependency or substitutability."
        ),
        "limitation": (
            "Large incoming trade value alone does not establish "
            "that a country depends on a particular exporter."
        ),
    })

    # 5. Flag incomplete-coverage caveat explicitly.
    insights.append({
        "category": "data_quality",
        "severity": "warning",
        "title": "Validate trade-data coverage",
        "finding": (
            "The ingestion run reported failed country requests. "
            "The network therefore should be treated as a partial "
            "view until coverage has been checked."
        ),
        "recommendation": (
            "Retry failed requests, record which countries were "
            "retrieved, and compare coverage before publishing "
            "country rankings as findings."
        ),
        "limitation": (
            "Missing data can distort centrality and trade-value "
            "rankings; missing edges must not be treated as zero trade."
        ),
    })

    return insights


def main():
    ml, network = load_inputs()
    insights = build_insights(ml, network)

    report = {
        "project": "Supply Chain Risk Intelligence Platform",
        "status": "exploratory",
        "sources": [
            str(ML_FILE.relative_to(PROJECT_ROOT)),
            str(NETWORK_FILE.relative_to(PROJECT_ROOT)),
        ],
        "insights": insights,
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "SUPPLY CHAIN RISK INTELLIGENCE REPORT",
        "=" * 48,
        "Status: Exploratory; validate data coverage before decisions.",
        "",
    ]

    for number, insight in enumerate(insights, start=1):
        lines.extend([
            f"{number}. {insight['title']}",
            f"Category: {insight['category']}",
            f"Severity: {insight['severity']}",
            f"Finding: {insight['finding']}",
            f"Recommendation: {insight['recommendation']}",
            f"Limitation: {insight['limitation']}",
            "",
        ])

    OUTPUT_TEXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nRISK INTELLIGENCE ENGINE COMPLETE")
    print(f"Insights generated: {len(insights)}")
    print(f"Text report: {OUTPUT_TEXT}")
    print(f"JSON report: {OUTPUT_JSON}")

    print("\nKEY FINDINGS")
    for insight in insights:
        print(f"\n- {insight['title']}")
        print(f"  {insight['finding']}")


if __name__ == "__main__":
    main()
