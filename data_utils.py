"""Shared data loading, feature engineering, and scoring utilities for the dashboard."""
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "global_supply_chain_disruption_v1.csv")
CLF_PATH = os.path.join(BASE_DIR, "delay_classifier.joblib")
REG_PATH = os.path.join(BASE_DIR, "delay_regressor.joblib")

FEATURES_NUMERIC = [
    "Base_Lead_Time_Days",
    "Scheduled_Lead_Time_Days",
    "Schedule_Buffer_Days",
    "Schedule_Buffer_Ratio",
    "Geopolitical_Risk_Index",
    "Weather_Severity_Index",
    "Inflation_Rate_Pct",
    "Shipping_Cost_USD",
    "Order_Weight_Kg",
    "Cost_Per_Kg",
    "External_Risk_Pressure",
    "Order_Month",
    "Order_Quarter",
    "Order_DayOfWeek",
]

FEATURES_CATEGORICAL = [
    "Origin_City",
    "Destination_City",
    "Route_Type",
    "Transportation_Mode",
    "Product_Category",
]

# Approximate coordinates for every port/city in the dataset (for the map view)
CITY_COORDS = {
    "Shanghai, CN": (31.23, 121.47),
    "Los Angeles, US": (34.05, -118.24),
    "Tokyo, JP": (35.68, 139.69),
    "Singapore, SG": (1.35, 103.82),
    "Shenzhen, CN": (22.54, 114.06),
    "Rotterdam, NL": (51.92, 4.48),
    "Santos, BR": (-23.96, -46.33),
    "Hamburg, DE": (53.55, 9.99),
    "New York, US": (40.71, -74.01),
    "Mumbai, IN": (19.08, 72.88),
    "Felixstowe, UK": (51.96, 1.35),
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Delay_Days"] = df["Delay_Days"].clip(lower=0)
    df["Is_Delayed"] = (df["Delivery_Status"].str.strip().str.lower() == "late").astype(int)

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_Quarter"] = df["Order_Date"].dt.quarter
    df["Order_Year"] = df["Order_Date"].dt.year
    df["Order_DayOfWeek"] = df["Order_Date"].dt.dayofweek

    df["Schedule_Buffer_Days"] = df["Scheduled_Lead_Time_Days"] - df["Base_Lead_Time_Days"]
    df["Schedule_Buffer_Ratio"] = (
        df["Schedule_Buffer_Days"] / df["Base_Lead_Time_Days"].replace(0, np.nan)
    ).fillna(0)

    df["Route"] = df["Origin_City"] + " -> " + df["Destination_City"]
    df["Cost_Per_Kg"] = df["Shipping_Cost_USD"] / df["Order_Weight_Kg"].replace(0, np.nan)
    df["Cost_Per_Kg"] = df["Cost_Per_Kg"].fillna(df["Cost_Per_Kg"].median())

    df["External_Risk_Pressure"] = (
        0.5 * df["Geopolitical_Risk_Index"] + 0.5 * (df["Weather_Severity_Index"] / 10)
    )
    return df


def load_models():
    for path in (CLF_PATH, REG_PATH):
        if not os.path.exists(path):
            present = os.listdir(BASE_DIR)
            raise FileNotFoundError(
                f"Could not find '{os.path.basename(path)}' in {BASE_DIR}.\n"
                f"Files actually present in that folder: {present}\n"
                "Same fix as the data file: check .gitignore isn't excluding .joblib files, "
                "confirm it's committed and pushed, then reboot the app."
            )
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg


def load_scored_data():
    if not os.path.exists(DATA_PATH):
        present = os.listdir(BASE_DIR)
        raise FileNotFoundError(
            f"Could not find '{os.path.basename(DATA_PATH)}' in {BASE_DIR}.\n"
            f"Files actually present in that folder: {present}\n"
            "This usually means the CSV was not committed to the repo, or is excluded "
            "by a .gitignore rule (check for a line like '*.csv' or 'data/'). "
            "Make sure the file sits in the SAME folder as app.py, commit it with "
            "`git add -f global_supply_chain_disruption_v1.csv` if it's gitignored, "
            "push, then reboot the app from Streamlit Cloud's 'Manage app' menu."
        )
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    clf, reg = load_models()
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]

    df["Predicted_Delay_Probability"] = clf.predict_proba(X)[:, 1]
    df["Predicted_Delay_Days"] = np.clip(reg.predict(X), 0, None)

    p_delay = df["Predicted_Delay_Probability"]
    mag_norm = (df["Predicted_Delay_Days"] / df["Predicted_Delay_Days"].quantile(0.99)).clip(0, 1)
    ext_norm = (df["External_Risk_Pressure"] / df["External_Risk_Pressure"].quantile(0.99)).clip(0, 1)

    df["Shipment_Risk_Score"] = ((0.5 * p_delay + 0.3 * mag_norm + 0.2 * ext_norm) * 100).round(1)
    df["Risk_Tier"] = pd.cut(
        df["Shipment_Risk_Score"],
        bins=[-0.01, 20, 40, 65, 100],
        labels=["Low", "Medium", "High", "Severe"],
    )

    lat_o = df["Origin_City"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    lon_o = df["Origin_City"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    lat_d = df["Destination_City"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    lon_d = df["Destination_City"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    df["Origin_Lat"], df["Origin_Lon"] = lat_o, lon_o
    df["Dest_Lat"], df["Dest_Lon"] = lat_d, lon_d

    return df


def _build_input_row(inputs: dict) -> pd.DataFrame:
    base_lead = inputs["base_lead_time"]
    sched_lead = inputs["scheduled_lead_time"]
    buffer_days = sched_lead - base_lead
    buffer_ratio = buffer_days / base_lead if base_lead else 0
    weight = max(inputs["order_weight"], 1)
    cost_per_kg = inputs["shipping_cost"] / weight
    ext_risk = 0.5 * inputs["geo_risk"] + 0.5 * (inputs["weather_risk"] / 10)

    return pd.DataFrame([{
        "Base_Lead_Time_Days": base_lead,
        "Scheduled_Lead_Time_Days": sched_lead,
        "Schedule_Buffer_Days": buffer_days,
        "Schedule_Buffer_Ratio": buffer_ratio,
        "Geopolitical_Risk_Index": inputs["geo_risk"],
        "Weather_Severity_Index": inputs["weather_risk"],
        "Inflation_Rate_Pct": inputs["inflation"],
        "Shipping_Cost_USD": inputs["shipping_cost"],
        "Order_Weight_Kg": weight,
        "Cost_Per_Kg": cost_per_kg,
        "External_Risk_Pressure": ext_risk,
        "Order_Month": inputs["order_month"],
        "Order_Quarter": (inputs["order_month"] - 1) // 3 + 1,
        "Order_DayOfWeek": inputs["day_of_week"],
        "Origin_City": inputs["origin"],
        "Destination_City": inputs["destination"],
        "Route_Type": inputs["route_type"],
        "Transportation_Mode": inputs["mode"],
        "Product_Category": inputs["category"],
    }])


def score_single_shipment(inputs: dict) -> dict:
    """Score a single hypothetical shipment defined by a dict of raw inputs."""
    clf, reg = load_models()
    row = _build_input_row(inputs)
    ext_risk = row["External_Risk_Pressure"].iloc[0]

    p_delay = float(clf.predict_proba(row[FEATURES_NUMERIC + FEATURES_CATEGORICAL])[:, 1][0])
    delay_days = float(max(reg.predict(row[FEATURES_NUMERIC + FEATURES_CATEGORICAL])[0], 0))

    # Use dataset-wide 99th percentile constants for normalization consistency
    mag_norm = min(delay_days / 12.0, 1.0)   # ~99th pct of Predicted_Delay_Days in training data
    ext_norm = min(ext_risk / 0.95, 1.0)     # ~99th pct of External_Risk_Pressure

    score = round((0.5 * p_delay + 0.3 * mag_norm + 0.2 * ext_norm) * 100, 1)
    if score < 20:
        tier = "Low"
    elif score < 40:
        tier = "Medium"
    elif score < 65:
        tier = "High"
    else:
        tier = "Severe"

    return {
        "probability": p_delay,
        "expected_delay_days": delay_days,
        "risk_score": score,
        "risk_tier": tier,
    }


# ---------------------------------------------------------------------------
# SHAP / Explainable AI utilities — computed live, no static image files needed
# ---------------------------------------------------------------------------

def clean_feature_name(name: str) -> str:
    """Turn 'num__Cost_Per_Kg' / 'cat__Product_Category_Textiles' into readable labels."""
    name = name.replace("num__", "").replace("cat__", "")
    return name.replace("_", " ")


def _transform(clf_pipe, X: pd.DataFrame) -> pd.DataFrame:
    preproc = clf_pipe.named_steps["preproc"]
    X_trans = preproc.transform(X[FEATURES_NUMERIC + FEATURES_CATEGORICAL])
    feat_names = preproc.get_feature_names_out()
    return pd.DataFrame(X_trans, columns=feat_names)


def compute_global_shap(df: pd.DataFrame, sample_size: int = 800, random_state: int = 42):
    """Return (shap_values, X_transformed_sample, feature_names) for the classifier,
    computed on a random sample of the dataset for speed."""
    import shap

    clf, _ = load_models()
    sample = df.sample(min(sample_size, len(df)), random_state=random_state)
    X_trans_df = _transform(clf, sample)

    explainer = shap.TreeExplainer(clf.named_steps["clf"])
    shap_values = explainer.shap_values(X_trans_df)
    return shap_values, X_trans_df


def compute_single_shap(inputs: dict):
    """Return (shap_values_row, feature_values_row, base_value) for one hypothetical shipment."""
    import shap

    clf, _ = load_models()
    row = _build_input_row(inputs)
    X_trans_df = _transform(clf, row)

    explainer = shap.TreeExplainer(clf.named_steps["clf"])
    shap_values = explainer.shap_values(X_trans_df)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0] if len(np.shape(base_value)) else float(base_value)

    return shap_values[0], X_trans_df.iloc[0], float(base_value)


# ---------------------------------------------------------------------------
# 1. ANOMALY DETECTION (unsupervised) — Isolation Forest
# Flags shipments whose feature combination is statistically unusual, distinct
# from "high risk" (a shipment can be low-risk but still an outlier — e.g. an
# oddly cheap Air shipment — which is exactly the kind of thing a fraud/quality
# review would want surfaced separately from the delay-risk score).
# ---------------------------------------------------------------------------

ANOMALY_NUMERIC = [
    "Base_Lead_Time_Days", "Cost_Per_Kg", "Order_Weight_Kg",
    "Geopolitical_Risk_Index", "Weather_Severity_Index", "Inflation_Rate_Pct",
    "Shipping_Cost_USD", "Schedule_Buffer_Days",
]


def _fit_anomaly_model(df: pd.DataFrame):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[ANOMALY_NUMERIC])
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X_scaled)
    return model, scaler


def compute_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add Anomaly_Score (higher = more unusual) and Is_Anomaly flag to df."""
    df = df.copy()
    model, scaler = _fit_anomaly_model(df)
    X_scaled = scaler.transform(df[ANOMALY_NUMERIC])
    # decision_function: higher = more normal. Flip sign so higher = more anomalous.
    df["Anomaly_Score"] = -model.decision_function(X_scaled)
    df["Is_Anomaly"] = model.predict(X_scaled) == -1
    return df, model, scaler


def score_single_anomaly(inputs: dict, model, scaler) -> dict:
    row = _build_input_row(inputs)
    X_scaled = scaler.transform(row[ANOMALY_NUMERIC])
    score = float(-model.decision_function(X_scaled)[0])
    is_anomaly = bool(model.predict(X_scaled)[0] == -1)
    return {"anomaly_score": score, "is_anomaly": is_anomaly}


# ---------------------------------------------------------------------------
# 2. RISK ARCHETYPE CLUSTERING (unsupervised) — K-Means + PCA
# Segments the portfolio into behavioral clusters (independent of the
# supervised risk score) so a portfolio manager can see natural groupings,
# e.g. "high-cost expedited air" vs "chronic Suez sea congestion."
# ---------------------------------------------------------------------------

CLUSTER_NUMERIC = [
    "Base_Lead_Time_Days", "Cost_Per_Kg", "Order_Weight_Kg",
    "Geopolitical_Risk_Index", "Weather_Severity_Index",
    "Shipping_Cost_USD", "Schedule_Buffer_Days",
]


def compute_clusters(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df = df.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_NUMERIC])

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df["PCA_1"], df["PCA_2"] = coords[:, 0], coords[:, 1]

    profile = (
        df.groupby("Cluster")
        .agg(
            size=("Cluster", "count"),
            avg_risk_score=("Shipment_Risk_Score", "mean"),
            delay_rate=("Is_Delayed", "mean"),
            avg_cost=("Shipping_Cost_USD", "mean"),
            avg_cost_per_kg=("Cost_Per_Kg", "mean"),
            dominant_mode=("Transportation_Mode", lambda s: s.mode().iat[0]),
            dominant_category=("Product_Category", lambda s: s.mode().iat[0]),
        )
        .reset_index()
        .sort_values("avg_risk_score", ascending=False)
    )

    def label_row(r):
        risk_word = "High-Risk" if r["avg_risk_score"] >= 40 else ("Elevated" if r["avg_risk_score"] >= 20 else "Low-Risk")
        cost_word = "High-Cost/kg" if r["avg_cost_per_kg"] >= 3 else "Low-Cost/kg"
        return f"{risk_word}: {r['dominant_mode']}, {cost_word}, mostly {r['dominant_category']}"

    profile["archetype_label"] = profile.apply(label_row, axis=1)
    return df, profile


# ---------------------------------------------------------------------------
# 3. MONTE CARLO PORTFOLIO EXPOSURE SIMULATION
# Treats each shipment's predicted delay probability as an independent
# Bernoulli trial and simulates the distribution of total delayed shipments
# and total financial exposure (shipping cost weighted by delay likelihood) —
# a lightweight, VaR-style tail-risk view of the filtered portfolio.
# ---------------------------------------------------------------------------

def monte_carlo_portfolio(df: pd.DataFrame, n_simulations: int = 2000, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    probs = df["Predicted_Delay_Probability"].values
    costs = df["Shipping_Cost_USD"].values
    n = len(df)

    if n == 0:
        return None

    draws = rng.random((n_simulations, n)) < probs[None, :]  # shape (sims, n_shipments)
    delayed_counts = draws.sum(axis=1)
    exposure = (draws * costs[None, :]).sum(axis=1)

    return {
        "delayed_counts": delayed_counts,
        "exposure": exposure,
        "expected_delayed": float(probs.sum()),
        "expected_exposure": float((probs * costs).sum()),
        "p95_delayed": float(np.percentile(delayed_counts, 95)),
        "p95_exposure": float(np.percentile(exposure, 95)),
        "p99_exposure": float(np.percentile(exposure, 99)),
    }


# ---------------------------------------------------------------------------
# 4. AUTO-GENERATED UNDERWRITING MEMO (template-driven NLG)
# Turns the model outputs + SHAP drivers into a plain-English risk memo —
# the kind of one-paragraph summary a credit analyst would actually write.
# ---------------------------------------------------------------------------

TIER_HISTORICAL_RATE = {"Low": "0.2%", "Medium": "4.0%", "High": "88.5%", "Severe": "99.2%"}


def generate_underwriting_memo(shipment_desc: str, result: dict, sv_row, feat_row, top_n: int = 3) -> str:
    exp = pd.DataFrame({"feature": feat_row.index, "shap_value": sv_row})
    exp["abs"] = exp["shap_value"].abs()
    top = exp.sort_values("abs", ascending=False).head(top_n)

    increasing = [clean_feature_name(f) for f in top.loc[top["shap_value"] > 0, "feature"]]
    decreasing = [clean_feature_name(f) for f in top.loc[top["shap_value"] < 0, "feature"]]

    tier = result["risk_tier"]
    hist_rate = TIER_HISTORICAL_RATE.get(tier, "n/a")

    parts = [
        f"Shipment ({shipment_desc}) is scored in the **{tier.upper()}** risk tier, "
        f"with a predicted delay probability of {result['probability']*100:.1f}% and an "
        f"expected delay of {result['expected_delay_days']:.1f} day(s). "
        f"Historically, shipments in the {tier} tier were delayed {hist_rate} of the time."
    ]

    if increasing:
        parts.append(f"Risk is primarily driven up by: {', '.join(increasing)}.")
    if decreasing:
        parts.append(f"Risk is partially offset by: {', '.join(decreasing)}.")

    if tier in ("High", "Severe"):
        parts.append(
            "Recommendation: apply a risk-adjusted collateral haircut or margin "
            "before extending trade finance against this shipment, and consider "
            "requiring proof of alternate routing contingency."
        )
    elif tier == "Medium":
        parts.append("Recommendation: standard terms with routine monitoring are likely adequate.")
    else:
        parts.append("Recommendation: standard underwriting terms are appropriate; no additional haircut indicated.")

    return " ".join(parts)
