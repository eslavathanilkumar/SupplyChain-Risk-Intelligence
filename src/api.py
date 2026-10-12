
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

app = FastAPI(
    title="Supply Chain Risk Intelligence API",
    description="API for supply chain risk analysis and GenAI reports.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Supply Chain Risk Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "project": "Supply Chain Risk Intelligence",
    }


def read_text_report(filename: str) -> str:
    path = REPORT_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {filename}",
        )

    return path.read_text(encoding="utf-8")


def read_csv_report(filename: str) -> pd.DataFrame:
    path = REPORT_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {filename}",
        )

    return pd.read_csv(path)


@app.get("/api/v1/model-comparison")
def model_comparison():
    df = read_csv_report("ml_model_comparison.csv")
    df = df.astype(object).where(pd.notna(df), None)

    return {
        "models": df.to_dict(orient="records"),
        "note": "Historical results do not guarantee future performance.",
    }


@app.get("/api/v1/network/top-countries")
def top_countries(limit: int = 10):
    if limit < 1 or limit > 50:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 50",
        )

    df = read_csv_report("country_network_metrics.csv")

    if "betweenness_centrality" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Missing betweenness_centrality column",
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
            "Centrality indicates network position, not proven "
            "disruption probability or economic impact. Trade data "
            "coverage may be incomplete."
        ),
        "countries": df.to_dict(orient="records"),
    }


@app.get("/api/v1/risk-report")
def risk_report():
    return {
        "status": "exploratory",
        "report": read_text_report("risk_intelligence_report.txt"),
    }


@app.get("/api/v1/network/summary")
def network_summary():
    return {
        "report": read_text_report("network_analysis_report.txt"),
    }


@app.get("/api/v1/genai-briefing")
def genai_briefing():
    return {
        "format": "markdown",
        "briefing": read_text_report("genai_risk_briefing.md"),
    }
