# GenAI Supply Chain Risk Intelligence Platform

An end-to-end data science and AI project that combines delivery-risk prediction, international trade network analysis, rule-based risk insights, and a Generative AI briefing interface.

## Project Overview

Supply chains can be vulnerable to shipping delays and disruptions in interconnected trade networks. This project explores historical supply-chain data and country-level trade relationships to identify patterns, evaluate delivery-risk models, and communicate findings through an interactive dashboard.

**Key capabilities**
- Data ingestion, quality checks, and preprocessing
- Exploratory data analysis and visualizations
- Machine learning for late-delivery risk classification
- Directed country-trade network analysis
- Risk intelligence reports generated from analytical results
- GenAI-generated risk briefings using the Groq API
- FastAPI backend for accessing reports
- Streamlit dashboard for interactive exploration

## Architecture

```text
Public Supply-Chain Data
          |
          v
 Data Ingestion & Quality Checks
          |
          v
 Exploratory Data Analysis
          |
          v
 Delivery-Risk ML Models
          |
          +----------------------+
          |                      |
          v                      v
 Country Trade Data       Risk Intelligence
          |                      |
          v                      v
 Network Analysis          GenAI Briefing
          |                      |
          +-----------+----------+
                      |
                      v
                FastAPI Backend
                      |
                      v
              Streamlit Dashboard
```

## Technology Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Network analysis:** NetworkX
- **Visualization:** Matplotlib, Seaborn, Plotly
- **API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **GenAI:** Groq API
- **Model persistence:** Joblib
- **Version control:** Git, GitHub

## Machine Learning Results

The project compares Logistic Regression and Random Forest models for `Late_delivery_risk`.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 69.38% | 84.05% | 54.56% | 66.17% | 0.7412 |
| Logistic Regression | 69.10% | 83.91% | 54.06% | 65.76% | 0.7373 |

Random Forest performed slightly better on the evaluation metrics shown. Its recall of approximately 54.6% means that a substantial portion of positive late-delivery-risk cases were missed in the evaluation set.

These are exploratory evaluation results, not evidence of production performance.

## Network Analysis

A directed country-level trade network is analyzed using trade-flow data. The network metrics include:

- In-degree and out-degree
- Trade-value in-strength and out-strength
- Degree centrality
- Betweenness centrality

Betweenness centrality helps identify countries that occupy potentially important positions in the modeled network. It does **not**, by itself, prove that a country will cause a disruption or quantify its economic impact.

Some public trade-data requests were incomplete. Missing trade flows must not be interpreted as zero trade, and aggregate partner labels require careful treatment.

## GenAI Risk Briefing

The GenAI component uses the Groq API to produce a concise briefing from the project's existing risk intelligence report.

The briefing is designed to summarize:
- Delivery-risk model findings
- Trade-network observations
- Priority areas for further investigation
- Data limitations
- Suggested next analytical steps

The assistant is instructed to use the supplied evidence and avoid inventing supplier relationships, disruption probabilities, or financial impacts.

A valid Groq API key is required to regenerate the briefing. Keep credentials in a local `.env` file and never commit that file to GitHub.

## API Endpoints

The FastAPI application exposes the following routes:

| Endpoint | Purpose |
|---|---|
| `GET /` | API root |
| `GET /health` | Health check |
| `GET /api/v1/model-comparison` | Model comparison results |
| `GET /api/v1/network/top-countries` | Top countries by network metrics |
| `GET /api/v1/risk-report` | Risk intelligence report |
| `GET /api/v1/network/summary` | Network summary |
| `GET /api/v1/genai-briefing` | GenAI briefing |

Interactive API documentation is available at `/docs` while the local API server is running.

## Project Structure

```text
SupplyChain-Risk-Intelligence/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── dashboard/
│   └── app.py
├── models/
├── reports/
├── src/
│   ├── api.py
│   ├── data_ingestion.py
│   ├── eda_analysis.py
│   ├── ml_model.py
│   ├── trade_data_ingestion.py
│   ├── network_analysis.py
│   ├── risk_insights.py
│   ├── genai_risk_assistant.py
│   └── validate_project.py
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone the repository

After publishing it to GitHub:

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd SupplyChain-Risk-Intelligence
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure GenAI credentials

Create a local `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Replace the placeholder with your own key. Never commit `.env`.

### 5. Run the API

```powershell
python -m uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

### 6. Run the dashboard

Open a second terminal, activate the same virtual environment, and run:

```powershell
python -m streamlit run dashboard/app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

### 7. Validate the project

With the API running, execute:

```powershell
python src/validate_project.py
```

Some checks require generated reports, saved model artifacts, and the running API. Follow the project's data-processing and training steps if those artifacts are not present.

## Limitations and Responsible Use

- The dataset and model evaluation are exploratory; further validation is required before operational use.
- The current Random Forest recall indicates that many positive