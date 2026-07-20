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
| `assets/` | Pre-rendered SHAP plots shown in the Explainability tab |

## Tabs

1. **Portfolio Overview** — KPIs, risk tier distribution, calibration check
2. **Global Risk Map** — trade lanes plotted geographically, colored by average risk score, sized by volume
3. **Risk Explorer** — scatter/pie views, sortable table of highest-risk shipments, CSV export
4. **Shipment Risk Scorer** — interactive form to score a hypothetical shipment before it ships (the underwriting use case)
5. **Model Explainability** — SHAP feature importance, with the honest finding that route/cost structure outweighs the continuous risk indices

## Notes

- All filters (sidebar) apply across every tab.
- The scorer tab calls the same two trained models used for the bulk-scored dataset — no separate logic path, so numbers are consistent everywhere in the dashboard.
- If you retrain the models (see the `code/` folder from the main project), just drop the new `.joblib` files in this folder with the same filenames.
