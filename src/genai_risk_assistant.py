
from pathlib import Path
import os

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
INPUT_FILE = REPORT_DIR / "risk_intelligence_report.txt"
OUTPUT_FILE = REPORT_DIR / "genai_risk_briefing.md"

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
).strip()

API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """
You are a supply-chain risk intelligence analyst.

Create a concise, evidence-based briefing for a logistics manager.

Use only the supplied report. Never invent data, suppliers,
relationships, probabilities, or financial impacts.

Explain that:
- Random Forest recall is 54.6% on the evaluation set.
- Betweenness measures network position, not proven disruption impact.
- The trade network may be incomplete because some API requests failed.
- The results are exploratory and not production-ready.

Use these Markdown sections:
1. Executive Summary
2. Delivery Risk Model
3. Trade Network Findings
4. Priority Investigations
5. Data Limitations
6. Recommended Next Steps
"""


def main():
    print("Starting GenAI Risk Assistant...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input report not found: {INPUT_FILE}. "
            "Run src/risk_insights.py first."
        )

    if not API_KEY or API_KEY == "put_your_actual_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check the .env file "
            "in your project root. Do not share your key."
        )

    source_report = INPUT_FILE.read_text(encoding="utf-8")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Create the briefing using this project report:\n\n"
                    + source_report
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1600,
    }

    print(f"Calling Groq model: {MODEL}")

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )

    if not response.ok:
        raise RuntimeError(
            f"Groq API error {response.status_code}: "
            f"{response.text[:1200]}"
        )

    data = response.json()
    briefing = data["choices"][0]["message"]["content"].strip()

    if not briefing:
        raise RuntimeError("The API returned an empty briefing.")

    OUTPUT_FILE.write_text(
        "# GenAI Supply Chain Risk Briefing\n\n"
        + briefing
        + "\n",
        encoding="utf-8",
    )

    print("\nGENAI RISK ASSISTANT COMPLETE")
    print(f"Briefing saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
