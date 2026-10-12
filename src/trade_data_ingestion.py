
from pathlib import Path
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "external"
OUTPUT_FILE = OUTPUT_DIR / "country_trade_flows_2023.csv"

API_URL = (
    "https://comtradeapi.un.org/"
    "public/v1/preview/C/A/HS"
)

# UN Comtrade numeric reporter codes for selected economies.
COUNTRIES = {
    "United States": 842,
    "China": 156,
    "India": 699,
    "Japan": 392,
    "Germany": 276,
    "United Kingdom": 826,
    "France": 251,
    "Italy": 380,
    "Canada": 124,
    "Mexico": 484,
    "Brazil": 76,
    "Australia": 36,
    "South Korea": 410,
    "Netherlands": 528,
    "Spain": 724,
    "Indonesia": 360,
    "Singapore": 702,
    "Saudi Arabia": 682,
    "United Arab Emirates": 784,
    "Switzerland": 756,
}


def fetch_country_exports(session, country_name, reporter_code):
    """Fetch reported annual exports for one reporting economy."""

    params = {
        "reporterCode": reporter_code,
        "period": "2023",
        "flowCode": "X",
        "cmdCode": "TOTAL",
        "maxRecords": 500,
        "includeDesc": "true",
    }

    response = session.get(
        API_URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))

    records = payload.get("data", [])

    if not records:
        print(f"WARNING: No data returned for {country_name}.")
        return []

    print(
        f"{country_name}: received "
        f"{len(records)} trade records"
    )

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    failed_countries = []

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "SupplyChainRiskIntelligence/1.0"
        })

        for country_name, reporter_code in COUNTRIES.items():
            try:
                records = fetch_country_exports(
                    session,
                    country_name,
                    reporter_code,
                )
                all_records.extend(records)

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
            ) as error:
                failed_countries.append(country_name)
                print(
                    f"WARNING: Failed for {country_name}: "
                    f"{error}"
                )

            # Avoid sending requests too rapidly.
            time.sleep(3)

    if not all_records:
        raise RuntimeError(
            "No trade records were retrieved. "
            "Check your internet connection and the "
            "UN Comtrade public API response."
        )

    df = pd.DataFrame(all_records)

    required_columns = [
        "reporterCode",
        "reporterDesc",
        "partnerCode",
        "partnerDesc",
        "period",
        "primaryValue",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The API response is missing expected columns: "
            + ", ".join(missing_columns)
            + ". Inspect the API response before continuing."
        )

    # Remove aggregate partners such as "World".
    # We need country-to-country links for the graph.
    df["reporterCode"] = pd.to_numeric(
        df["reporterCode"], errors="coerce"
    )
    df["partnerCode"] = pd.to_numeric(
        df["partnerCode"], errors="coerce"
    )
    df["primaryValue"] = pd.to_numeric(
        df["primaryValue"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "reporterCode",
            "partnerCode",
            "primaryValue",
        ]
    )

    df = df[
        (df["partnerCode"] != 0)
        & (df["reporterCode"] != df["partnerCode"])
        & (df["primaryValue"] > 0)
    ].copy()

    # Keep only the fields needed for network analysis.
    output_columns = [
        "reporterCode",
        "reporterDesc",
        "partnerCode",
        "partnerDesc",
        "period",
        "primaryValue",
    ]

    df = df[output_columns].drop_duplicates()

    df = df.rename(columns={
        "reporterCode": "exporter_code",
        "reporterDesc": "exporter",
        "partnerCode": "partner_code",
        "partnerDesc": "partner",
        "period": "year",
        "primaryValue": "trade_value_usd",
    })

    df["year"] = df["year"].astype(str)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nTRADE DATA INGESTION COMPLETE")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Trade records saved: {len(df):,}")
    print(f"Exporting countries represented: "
          f"{df['exporter'].nunique()}")
    print(f"Partner countries represented: "
          f"{df['partner'].nunique()}")

    if failed_countries:
        print(
            "\nCountries with failed requests: "
            + ", ".join(failed_countries)
        )

    print("\nSample records:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
