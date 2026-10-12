
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

# Create FastAPI application
app = FastAPI(
    title="Supply Chain Risk Intelligence API",
    description=(
        "API for delivery-risk model results, trade-network "
        "analysis, and GenAI risk briefings."
    ),
    version="1.0.0",
)

# CORS configuration for local dashboard development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_text_report(filename: str) -> str:
    """Read a text or Markdown report from the reports folder."""
    path = REPORT_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {filename}. Run the relevant analysis first.",
        )

    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read report: {error}",
        ) from error


def read_csv_report(filename: str) -> pd.DataFrame:
    """Read a CSV report and handle missing files."""
    path = REPORT_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {filename}. Run the relevant analysis first.",
        )

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read CSV report: {error}",
        ) from error


@app.get("/")
def home():
    """API welcome endpoint."""
    return {
        "message": "Supply Chain Risk Intelligence API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health():
    """Check API availability."""
    return {
        "status": "healthy",
        "project": "Supply Chain Risk Intelligence",
    }


@app.get("/api/v1/model-comparison")
def model_comparison():
    """Return delivery-risk model evaluation metrics."""
    df = read_csv_report("ml_model_comparison.csv")
    df = df.astype(object).where(pd.notna(df), None)

    return {
        "models": df.to_dict(orient="records"),
        "note": (
            "Historical evaluation results do not guarantee "
            "future model performance."
        ),
    }


@app.get("/api/v1/network/top-countries")
def top_countries(limit: int = 10):
    """Return countries ranked by betweenness centrality."""
    if limit < 1 or limit > 50:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 50.",
        )

    df = read_csv_report("country_network_metrics.csv")

    if "betweenness_centrality" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=(
                "The network report is missing the "
                "betweenness_centrality column."
            ),
        )

    df["betweenness_centrality"] = pd.to_numeric(
        df["betweenness_centrality"],
        errors="coerce",
    )

    df = df.dropna(subset=["betweenness_centrality"])
    df = df.nlargest(limit, "betweenness_centrality")
    df = df.astype(object).where(pd.notna(df), None)

    return {
        "metric": "betweenness_centrality",
        "warning": (
            "Centrality measures network position, not confirmed "
            "disruption probability or economic impact. Trade-data "
            "coverage may be incomplete."
        ),
        "countries": df.to_dict(orient="records"),
    }


@app.get("/api/v1/risk-report")
def risk_report():
    """Return the rule-based risk intelligence report."""
    return {
        "status": "exploratory",
        "report": read_text_report("risk_intelligence_report.txt"),
    }


@app.get("/api/v1/network/summary")
def network_summary():
    """Return the country network analysis report."""
    return {
        "report": read_text_report("network_analysis_report.txt"),
    }


@app.get("/api/v1/genai-briefing")
def genai_briefing():
    """Return the generated GenAI risk briefing."""
    return {
        "format": "markdown",
        "briefing": read_text_report("genai_risk_briefing.md"),
    }
