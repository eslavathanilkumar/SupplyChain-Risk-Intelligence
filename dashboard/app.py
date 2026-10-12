
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

st.set_page_config(
    page_title="Supply Chain Risk Intelligence",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Supply Chain Risk Intelligence")
st.caption(
    "Exploratory analytics | Delivery-risk ML | Trade-network analysis | GenAI"
)

st.warning(
    "Exploratory project: trade data coverage may be incomplete. "
    "Network centrality is not proof of disruption probability or economic impact."
)


def load_csv(filename):
    path = REPORT_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not read {filename}: {error}")
        return None


def load_text(filename):
    path = REPORT_DIR / filename
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        st.error(f"Could not read {filename}: {error}")
        return None


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Executive Overview",
        "Delivery Risk Models",
        "Trade Network",
        "Risk Intelligence",
        "GenAI Briefing",
        "Project Limitations",
    ],
)

model_df = load_csv("ml_model_comparison.csv")
network_df = load_csv("country_network_metrics.csv")
risk_report = load_text("risk_intelligence_report.txt")
network_report = load_text("network_analysis_report.txt")
genai_report = load_text("genai_risk_briefing.md")


if page == "Executive Overview":
    st.header("Executive Overview")

    if model_df is not None and not model_df.empty:
        best = model_df.sort_values("ROC-AUC", ascending=False).iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Models evaluated", len(model_df))
        col2.metric("Best model", str(best["Model"]))
        col3.metric("Best ROC-AUC", f"{best['ROC-AUC']:.3f}")

        st.subheader("Model comparison")
        st.dataframe(model_df, use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison report not found.")

    if network_df is not None and not network_df.empty:
        centrality_col = "betweenness_centrality"

        if centrality_col in network_df.columns:
            network_df[centrality_col] = pd.to_numeric(
                network_df[centrality_col], errors="coerce"
            )
            valid = network_df.dropna(subset=[centrality_col])

            if not valid.empty:
                top = valid.nlargest(5, centrality_col)
                st.subheader("Top countries by network centrality")
                st.dataframe(top, use_container_width=True, hide_index=True)

    st.info(
        "Use the sidebar to explore model evaluation, trade-network metrics, "
        "and generated risk briefings."
    )


elif page == "Delivery Risk Models":
    st.header("Delivery Risk Model Evaluation")

    if model_df is None or model_df.empty:
        st.error("Missing reports/ml_model_comparison.csv")
    else:
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        metrics = [
            column
            for column in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            if column in model_df.columns
        ]

        if metrics and "Model" in model_df.columns:
            metric = st.selectbox("Choose a metric", metrics)
            chart = px.bar(
                model_df,
                x="Model",
                y=metric,
                title=f"Model comparison: {metric}",
                text_auto=".3f",
                range_y=[0, 1],
            )
            st.plotly_chart(chart, use_container_width=True)

        st.warning(
            "Random Forest recall was approximately 54.6% in the recorded "
            "evaluation. Some positive cases were missed. Validate thresholds "
            "and test on newer data before operational use."
        )


elif page == "Trade Network":
    st.header("Country Trade Network")

    if network_df is None or network_df.empty:
        st.error("Missing reports/country_network_metrics.csv")
    elif "betweenness_centrality" not in network_df.columns:
        st.error("The network metrics file has no betweenness_centrality column.")
    else:
        network_df["betweenness_centrality"] = pd.to_numeric(
            network_df["betweenness_centrality"], errors="coerce"
        )
        valid = network_df.dropna(subset=["betweenness_centrality"])

        name_col = next(
            (
                col for col in ["country", "Country", "name", "node"]
                if col in valid.columns
            ),
            valid.columns[0],
        )

        limit = st.slider("Countries to display", 5, 30, 10)
        top = valid.nlargest(limit, "betweenness_centrality")

        chart = px.bar(
            top.sort_values("betweenness_centrality"),
            x="betweenness_centrality",
            y=name_col,
            orientation="h",
            title="Countries ranked by betweenness centrality",
            labels={
                "betweenness_centrality": "Betweenness centrality",
                name_col: "Country / node",
            },
        )
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(top, use_container_width=True, hide_index=True)

    with st.expander("Network analysis report"):
        if network_report:
            st.text(network_report)
        else:
            st.info("Network analysis report not found.")


elif page == "Risk Intelligence":
    st.header("Rule-Based Risk Intelligence")

    if risk_report:
        st.text(risk_report)
    else:
        st.error("Missing reports/risk_intelligence_report.txt")


elif page == "GenAI Briefing":
    st.header("GenAI Supply Chain Risk Briefing")

    if genai_report:
        st.markdown(genai_report)
    else:
        st.info(
            "GenAI briefing not found. Generate it first by running "
            "`python .\\src\\genai_risk_assistant.py` from the project root."
        )


elif page == "Project Limitations":
    st.header("Data Quality and Project Limitations")

    st.markdown(
        """
        - Trade API requests failed for some economies, so network coverage
          is incomplete.
        - Missing trade records must not be interpreted as zero trade.
        - Aggregate labels such as `Areas, nes` are not ordinary countries.
        - Betweenness centrality measures structural network position, not
          confirmed disruption likelihood or economic impact.
        - ML evaluation results describe the available historical data and
          do not guarantee future performance.
        - Recall of approximately 54.6% means the evaluated model missed
          some positive cases.
        - All findings should be validated before operational decisions.
        """
    )
