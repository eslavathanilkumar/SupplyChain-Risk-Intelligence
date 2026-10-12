# GenAI Supply Chain Risk Briefing

## 1. Executive Summary
- The analysis is **exploratory**; data coverage must be validated before any operational decisions.  
- A Random Forest model predicts delivery‑risk with **ROC‑AUC 0.741** and **recall 54.6 %** on the evaluation set.  
- Network‑centrality results highlight countries that sit in key positions (high betweenness) and those with the largest reported trade flows, but **betweenness is a structural metric only** – it does not prove a disruption impact.  
- The trade graph is **incomplete** because several API requests failed, so rankings may be biased.  

---

## 2. Delivery Risk Model
- **Model:** Random Forest (best by ROC‑AUC).  
- **Performance on evaluation set:**  
  - ROC‑AUC: **0.741**  
  - F1 score: **0.662**  
  - **Recall: 54.6 %** (i.e., the model correctly identifies just over half of the risky deliveries).  
- **Implication:** Before using predictions to prioritize interventions, compare recall and precision across different classification thresholds.  
- **Caveat:** Metrics reflect the historical dataset only; they do **not guarantee future performance**.

---

## 3. Trade Network Findings
| Aspect | Finding (from current graph) |
|--------|------------------------------|
| **High betweenness centrality** (structural position) | China, Netherlands, Spain, Japan, United Kingdom |
| **Largest summed outgoing trade values** | China, USA, Germany, Netherlands, Japan |
| **Largest summed incoming trade values** | USA, China, Germany, *Areas, nes*, Mexico |

- **Betweenness** tells us which nodes sit on many shortest paths; it **does not prove economic impact or disruption probability**.  
- Outgoing and incoming trade values identify where large reported flows exist, but **they are not definitive evidence of dependency**.

---

## 4. Priority Investigations
1. **High‑betweenness countries** – map the specific trade connections for China, Netherlands, Spain, Japan, and the UK; simulate removal of each node to see how network connectivity changes.  
2. **Trade concentration** – drill into the edge list for the top outgoing exporters (China, USA, Germany, Netherlands, Japan) to pinpoint critical partners, product categories, or regions.  
3. **Incoming flow analysis** – examine the partners and product types behind the largest inbound values (USA, China, Germany, etc.) before drawing conclusions about reliance.  

---

## 5. Data Limitations
- **Incomplete network:** The ingestion run reported failed country requests; the graph should be treated as a **partial view** until coverage is re‑checked.  
- **Missing edges ≠ zero trade:** Absence of data must not be interpreted as no trade; it can distort centrality and trade‑value rankings.  
- **Model scope:** Performance metrics are tied to the historical dataset and **do not guarantee future predictive power**.  
- Overall, the findings are **exploratory and not production‑ready**.

---

## 6. Recommended Next Steps
1. **Validate data coverage** – retry all failed API requests, log which countries are successfully retrieved, and quantify coverage gaps.  
2. **Re‑build the trade graph** with the complete data set; recompute centrality and trade‑value rankings.  
3. **Threshold analysis** for the Random Forest model – plot recall vs. precision across thresholds to select an operating point aligned with operational risk tolerance.  
4. **Scenario testing** – remove each high‑betweenness node in the updated graph to assess potential disruption cascades.  
5. **Enrich the dataset** – where possible, supplement API data with partner‑reported import figures to reduce asymmetry.  
6. **Pilot implementation** – once data quality and model thresholds are validated, run a limited‑scope pilot to compare model alerts with real‑world delivery outcomes before full deployment.
