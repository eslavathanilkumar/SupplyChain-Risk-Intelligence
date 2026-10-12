
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "country_trade_flows_2023.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

METRICS_FILE = REPORT_DIR / "country_network_metrics.csv"
EDGES_FILE = REPORT_DIR / "country_network_edges.csv"
REPORT_FILE = REPORT_DIR / "network_analysis_report.txt"
GRAPH_FILE = FIGURE_DIR / "country_trade_network.png"


def load_trade_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Trade data not found: {DATA_FILE}\n"
            "Run src/trade_data_ingestion.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "exporter",
        "partner",
        "trade_value_usd",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["trade_value_usd"] = pd.to_numeric(
        df["trade_value_usd"], errors="coerce"
    )

    df = df.dropna(
        subset=["exporter", "partner", "trade_value_usd"]
    )

    df = df[
        (df["exporter"] != df["partner"])
        & (df["trade_value_usd"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid trade-flow records found.")

    return df


def build_network(df):
    # Multiple records for the same directed country pair are
    # combined into one edge, summing their reported trade values.
    edges = (
        df.groupby(["exporter", "partner"], as_index=False)
        ["trade_value_usd"]
        .sum()
    )

    graph = nx.DiGraph()

    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.exporter,
            row.partner,
            trade_value_usd=float(row.trade_value_usd),
        )

    return graph, edges


def calculate_metrics(graph):
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())

    in_strength = dict(
        graph.in_degree(weight="trade_value_usd")
    )
    out_strength = dict(
        graph.out_degree(weight="trade_value_usd")
    )

    degree_centrality = nx.degree_centrality(graph)

    # Betweenness is calculated on the directed graph.
    # It measures structural brokerage, not actual disruption impact.
    betweenness = nx.betweenness_centrality(
        graph, normalized=True
    )

    rows = []

    for country in graph.nodes():
        rows.append({
            "country": country,
            "in_degree": in_degree[country],
            "out_degree": out_degree[country],
            "in_strength_trade_value_usd": in_strength[country],
            "out_strength_trade_value_usd": out_strength[country],
            "degree_centrality": degree_centrality[country],
            "betweenness_centrality": betweenness[country],
        })

    metrics = pd.DataFrame(rows)

    return metrics.sort_values(
        "betweenness_centrality", ascending=False
    )


def draw_network(graph):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Spring layout is useful for this exploratory graph.
    # It does not represent geographic locations.
    plt.figure(figsize=(18, 13))

    positions = nx.spring_layout(
        graph,
        seed=42,
        k=1.2,
        iterations=80,
    )

    node_sizes = [
        120 + 20 * graph.degree(node)
        for node in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        alpha=0.8,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=8,
        alpha=0.18,
        width=0.6,
        connectionstyle="arc3,rad=0.05",
    )

    # Label only the most connected countries to reduce clutter.
    top_nodes = sorted(
        graph.degree(),
        key=lambda item: item[1],
        reverse=True,
    )[:25]

    labels = {node: node for node, _ in top_nodes}

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8,
    )

    plt.title(
        "Reported Country-to-Country Trade Network (2023)\n"
        "Directed edges represent reported export flows"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_trade_data()
    graph, edges = build_network(df)
    metrics = calculate_metrics(graph)

    edges.to_csv(EDGES_FILE, index=False)
    metrics.to_csv(METRICS_FILE, index=False)

    draw_network(graph)

    top_betweenness = metrics.head(10)
    top_exports = metrics.sort_values(
        "out_strength_trade_value_usd", ascending=False
    ).head(10)
    top_imports = metrics.sort_values(
        "in_strength_trade_value_usd", ascending=False
    ).head(10)

    lines = [
        "COUNTRY TRADE NETWORK ANALYSIS",
        "=" * 40,
        "Year: 2023",
        "Source: UN Comtrade public preview API",
        "Interpretation: reported country-level trade flows",
        "",
        f"Input records: {len(df):,}",
        f"Graph nodes: {graph.number_of_nodes():,}",
        f"Directed edges: {graph.number_of_edges():,}",
        f"Weakly connected components: "
        f"{nx.number_weakly_connected_components(graph):,}",
        "",
        "TOP COUNTRIES BY BETWEENNESS CENTRALITY",
        "-" * 40,
        top_betweenness[
            ["country", "betweenness_centrality"]
        ].to_string(index=False),
        "",
        "TOP COUNTRIES BY REPORTED OUTGOING TRADE VALUE",
        "-" * 40,
        top_exports[
            ["country", "out_strength_trade_value_usd"]
        ].to_string(index=False),
        "",
        "TOP COUNTRIES BY REPORTED INCOMING TRADE VALUE",
        "-" * 40,
        top_imports[
            ["country", "in_strength_trade_value_usd"]
        ].to_string(index=False),
        "",
        "LIMITATIONS",
        "-" * 40,
        "The graph contains only the records retrieved by the ingestion script.",
        "Missing countries or edges do not imply zero trade.",
        "Reported exports may differ from partner-reported imports.",
        "Trade values do not establish direct supplier relationships.",
        "Betweenness indicates structural brokerage, not proven disruption impact.",
    ]

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nNETWORK ANALYSIS COMPLETE")
    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Directed edges: {graph.number_of_edges():,}")
    print(f"Metrics: {METRICS_FILE}")
    print(f"Edge list: {EDGES_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"Graph image: {GRAPH_FILE}")

    print("\nTop 10 countries by betweenness:")
    print(
        top_betweenness[
            ["country", "betweenness_centrality"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
