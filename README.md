# Supply Chain Disruption Risk Engine — Dashboard

Interactive Streamlit dashboard for the leading-indicator shipment risk model
built on the `global_supply_chain_disruption_v1.csv` dataset.

## Setup

```bash
cd supply_chain_risk_dashboard
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Main dashboard — 5 tabs (Overview, Map, Explorer, Scorer, Explainability) |
| `data_utils.py` | Feature engineering, model loading, scoring logic — mirrors the training pipeline exactly so live scores match the offline model |
| `delay_classifier.joblib` | Trained XGBoost delay-probability classifier |
| `delay_regressor.joblib` | Trained XGBoost expected-delay-days regressor |
| `global_supply_chain_disruption_v1.csv` | Source dataset (10,000 shipments) |

## Tabs

1. **Portfolio Overview** — KPIs, risk tier distribution, calibration check
2. **Global Risk Map** — trade lanes plotted geographically, colored by average risk score, sized by volume
3. **Risk Explorer** — scatter/pie views, sortable table of highest-risk shipments, CSV export
4. **Shipment Risk Scorer** — interactive form to score a hypothetical shipment before it ships, including a live per-shipment SHAP explanation of *why* it scored that way
5. **Model Explainability** — SHAP feature importance and beeswarm plots, computed live from the loaded model (no static images to go stale or go missing) — with the honest finding that route/cost structure outweighs the continuous risk indices
6. **Advanced AI Lab** — three additional ML techniques distinct from the core classifier/regressor:
   - **Anomaly Detection** (Isolation Forest, unsupervised): flags shipments with statistically unusual cost/weight/lead-time profiles, independent of the risk score — a shipment can be Low-risk-tier and still get flagged as an outlier
   - **Risk Archetypes** (K-Means + PCA, unsupervised): segments the portfolio into natural clusters (e.g. "high-cost/kg Air Semiconductors, elevated risk" vs "low-cost Sea Raw Materials, low risk") without using the delay label at all
   - **Portfolio Monte Carlo Simulation**: treats each shipment's predicted delay probability as an independent trial and simulates thousands of portfolio outcomes to produce expected and 95th/99th-percentile tail-risk figures for delayed-shipment count and financial exposure — a lightweight VaR-style view

The Scorer tab also now includes an anomaly check and an auto-generated plain-English underwriting memo for every shipment you score, built from the model outputs and SHAP drivers (template-based NLG, not a hosted LLM call — no API key needed).

## Deep Learning Model (PyTorch)

`dl_model.py` trains a genuine deep learning alternative to the XGBoost classifier — a feedforward neural net with learned embeddings for the categorical features (Origin, Destination, Route Type, Mode, Category), concatenated with standardized numeric features and passed through a 3-layer MLP. Trained on the same leading-indicator features and the same train/test split as XGBoost, so results are directly comparable (XGBoost: ROC-AUC 0.970, PyTorch DNN: ROC-AUC 0.952 — trees have a modest edge here, which is itself a fair, honest finding rather than something to paper over).

The Scorer tab shows both models' predictions side by side with an **agreement indicator**. Large disagreement between the two independently-trained models is flagged as a signal worth a manual review, rather than blindly trusting either number — this genuinely caught an edge case during testing where an input sat outside what a specific route/mode/category combination had ever seen in training.

**Model files are pre-trained and included** (`deep_risk_net.pt`, `deep_risk_net_meta.json`). If you ever want to retrain: `python dl_model.py` regenerates both files from the CSV.

## Live Port Conditions (Open-Meteo)

The **🛰️ Live Port Conditions** tab pulls real, live marine weather (current + 7-day hourly wave height, wind speed) for each of the 11 ports from **Open-Meteo** — free, no API key required. This replaces the dataset's static `Weather_Severity_Index` with an actual live feed.

- **Auto-refresh toggle** re-fetches on an interval (1–30 min, your choice) using `streamlit-autorefresh`
- **Manual "Refresh Now" button** clears the cache and re-fetches immediately
- Results are cached for 10 minutes by default to avoid hammering the API between refreshes
- If the API call fails (network issue, rate limit), the tab shows a clear error instead of crashing — the rest of the dashboard keeps working
- A "suggested Weather Severity Index" is computed from live conditions so you can carry a real reading into the Scorer tab's slider

**Important:** this was built and tested against Open-Meteo's documented request/response format, but the dev environment it was built in has restricted network egress and could not reach `open-meteo.com` to do a live end-to-end test. Streamlit Cloud has open internet access, so verify this tab works correctly right after deploying — if it doesn't, check the error message shown (it will tell you exactly what failed).

## Notes

- All filters (sidebar) apply across every tab.
- The scorer tab calls the same two trained models used for the bulk-scored dataset — no separate logic path, so numbers are consistent everywhere in the dashboard.
- If you retrain the models (see the `code/` folder from the main project), just drop the new `.joblib` files in this folder with the same filenames.
