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
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg


def load_scored_data():
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


def score_single_shipment(inputs: dict) -> dict:
    """Score a single hypothetical shipment defined by a dict of raw inputs."""
    clf, reg = load_models()

    base_lead = inputs["base_lead_time"]
    sched_lead = inputs["scheduled_lead_time"]
    buffer_days = sched_lead - base_lead
    buffer_ratio = buffer_days / base_lead if base_lead else 0
    weight = max(inputs["order_weight"], 1)
    cost_per_kg = inputs["shipping_cost"] / weight
    ext_risk = 0.5 * inputs["geo_risk"] + 0.5 * (inputs["weather_risk"] / 10)

    row = pd.DataFrame([{
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
