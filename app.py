"""
Supply Chain Disruption Risk Engine — Interactive Dashboard
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_utils import load_scored_data, score_single_shipment, CITY_COORDS

st.set_page_config(
    page_title="Supply Chain Risk Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / CSS — dark intel-dashboard aesthetic
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg-primary: #0b0f19;
    --bg-panel: #111827;
    --bg-card: #151b2c;
    --accent-teal: #2dd4bf;
    --accent-amber: #f5b942;
    --accent-red: #ef4444;
    --text-primary: #e5e7eb;
    --text-muted: #8b93a7;
    --border-color: #232b3e;
}
.stApp {
    background: radial-gradient(circle at 20% 0%, #0f1626 0%, #0b0f19 60%);
    color: var(--text-primary);
}
h1, h2, h3 { font-family: 'Segoe UI', sans-serif; letter-spacing: 0.5px; }
h1 { color: var(--accent-teal) !important; }
h2, h3 { color: #dfe4ee !important; }
[data-testid="stSidebar"] {
    background-color: var(--bg-panel);
    border-right: 1px solid var(--border-color);
}
div[data-testid="stMetric"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
div[data-testid="stMetricValue"] { color: var(--accent-teal) !important; }
.risk-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-weight: 600; font-size: 0.85rem; letter-spacing: 0.5px;
}
.badge-low { background: rgba(45, 212, 191, 0.15); color: #2dd4bf; border: 1px solid #2dd4bf; }
.badge-medium { background: rgba(245, 185, 66, 0.15); color: #f5b942; border: 1px solid #f5b942; }
.badge-high { background: rgba(255, 140, 66, 0.15); color: #ff8c42; border: 1px solid #ff8c42; }
.badge-severe { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444; }
.header-strip {
    padding: 18px 22px; border-radius: 12px;
    background: linear-gradient(90deg, #101828 0%, #0d1420 100%);
    border: 1px solid var(--border-color); margin-bottom: 18px;
}
.small-caption { color: var(--text-muted); font-size: 0.85rem; }
hr { border-color: var(--border-color) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TIER_COLORS = {"Low": "#2dd4bf", "Medium": "#f5b942", "High": "#ff8c42", "Severe": "#ef4444"}
BADGE_CLASS = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high", "Severe": "badge-severe"}


@st.cache_data(show_spinner="Loading shipment data and scoring model...")
def get_data():
    return load_scored_data()


df = get_data()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-strip">
        <h1 style="margin-bottom:0;">🛰️ Supply Chain Disruption Risk Engine</h1>
        <span class="small-caption">Leading-indicator risk scoring for trade finance &amp; supply-chain-finance underwriting · 10,000 shipments · XGBoost + SHAP</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Filters")
modes = st.sidebar.multiselect("Transportation Mode", sorted(df["Transportation_Mode"].unique()), default=list(df["Transportation_Mode"].unique()))
route_types = st.sidebar.multiselect("Route Type", sorted(df["Route_Type"].unique()), default=list(df["Route_Type"].unique()))
categories = st.sidebar.multiselect("Product Category", sorted(df["Product_Category"].unique()), default=list(df["Product_Category"].unique()))
tiers = st.sidebar.multiselect("Risk Tier", ["Low", "Medium", "High", "Severe"], default=["Low", "Medium", "High", "Severe"])

fdf = df[
    df["Transportation_Mode"].isin(modes)
    & df["Route_Type"].isin(route_types)
    & df["Product_Category"].isin(categories)
    & df["Risk_Tier"].astype(str).isin(tiers)
]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<span class='small-caption'>Showing **{len(fdf):,}** of {len(df):,} shipments</span>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_overview, tab_map, tab_explorer, tab_scorer, tab_explain = st.tabs(
    ["📊 Portfolio Overview", "🌍 Global Risk Map", "🔍 Risk Explorer", "🎯 Shipment Risk Scorer", "🧠 Model Explainability"]
)

# ============================== OVERVIEW ==================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shipments (filtered)", f"{len(fdf):,}")
    c2.metric("Actual Delay Rate", f"{fdf['Is_Delayed'].mean()*100:.1f}%")
    c3.metric("Avg Risk Score", f"{fdf['Shipment_Risk_Score'].mean():.1f}")
    c4.metric("High + Severe Tier", f"{(fdf['Risk_Tier'].isin(['High','Severe'])).mean()*100:.1f}%")

    st.markdown("#### Risk Tier Distribution")
    tier_counts = fdf["Risk_Tier"].value_counts().reindex(["Low", "Medium", "High", "Severe"]).fillna(0)
    fig = go.Figure(go.Bar(
        x=tier_counts.index, y=tier_counts.values,
        marker_color=[TIER_COLORS[t] for t in tier_counts.index],
        text=tier_counts.values, textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
        height=350, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Delay Rate by Transportation Mode")
        rate_mode = fdf.groupby("Transportation_Mode")["Is_Delayed"].mean().sort_values(ascending=False)
        fig = px.bar(x=rate_mode.values, y=rate_mode.index, orientation="h",
                      color=rate_mode.values, color_continuous_scale="Teal")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                           height=320, showlegend=False, coloraxis_showscale=False,
                           xaxis_title="Delay Rate", yaxis_title="")
        st.plotly_chart(fig, width='stretch')
    with col_b:
        st.markdown("#### Delay Rate by Product Category")
        rate_cat = fdf.groupby("Product_Category")["Is_Delayed"].mean().sort_values(ascending=False)
        fig = px.bar(x=rate_cat.values, y=rate_cat.index, orientation="h",
                      color=rate_cat.values, color_continuous_scale="Oranges")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                           height=320, showlegend=False, coloraxis_showscale=False,
                           xaxis_title="Delay Rate", yaxis_title="")
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Risk Tier Calibration Check")
    calib = fdf.groupby("Risk_Tier")["Is_Delayed"].mean().reindex(["Low", "Medium", "High", "Severe"])
    fig = go.Figure(go.Bar(
        x=calib.index, y=calib.values * 100,
        marker_color=[TIER_COLORS[t] for t in calib.index],
        text=[f"{v*100:.1f}%" for v in calib.values], textposition="outside",
    ))
    fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                       height=320, yaxis_title="Actual Delay Rate (%)", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')
    st.caption("Each tier's actual delay rate should increase monotonically — confirming the score is well-calibrated, not just confident.")

# ============================== MAP ========================================
with tab_map:
    st.markdown("#### Trade Lane Risk Map")
    st.caption("Line color = average predicted risk score on that lane · Line width = shipment volume")

    lane_agg = (
        fdf.groupby(["Origin_City", "Destination_City"])
        .agg(
            avg_score=("Shipment_Risk_Score", "mean"),
            volume=("Shipment_Risk_Score", "count"),
            delay_rate=("Is_Delayed", "mean"),
        )
        .reset_index()
    )

    fig = go.Figure()
    max_vol = lane_agg["volume"].max() if len(lane_agg) else 1
    for _, row in lane_agg.iterrows():
        o = CITY_COORDS.get(row["Origin_City"])
        d = CITY_COORDS.get(row["Destination_City"])
        if not o or not d:
            continue
        score = row["avg_score"]
        color = "#2dd4bf" if score < 20 else "#f5b942" if score < 40 else "#ff8c42" if score < 65 else "#ef4444"
        fig.add_trace(go.Scattergeo(
            lon=[o[1], d[1]], lat=[o[0], d[0]],
            mode="lines",
            line=dict(width=1.5 + 6 * row["volume"] / max_vol, color=color),
            opacity=0.75,
            hoverinfo="text",
            text=f"{row['Origin_City']} → {row['Destination_City']}<br>Avg risk score: {score:.1f}<br>Delay rate: {row['delay_rate']*100:.1f}%<br>Volume: {row['volume']}",
            showlegend=False,
        ))

    city_list = list(CITY_COORDS.keys())
    fig.add_trace(go.Scattergeo(
        lon=[CITY_COORDS[c][1] for c in city_list],
        lat=[CITY_COORDS[c][0] for c in city_list],
        mode="markers+text",
        text=city_list, textposition="top center",
        textfont=dict(color="#e5e7eb", size=10),
        marker=dict(size=7, color="#e5e7eb", line=dict(width=1, color="#0b0f19")),
        showlegend=False,
        hoverinfo="text",
    ))

    fig.update_geos(
        projection_type="natural earth",
        bgcolor="#0b0f19", landcolor="#1a2236", oceancolor="#0b0f19",
        showcountries=True, countrycolor="#232b3e", showland=True,
        lakecolor="#0b0f19",
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0b0f19",
        height=560, margin=dict(t=10, b=10, l=0, r=0),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("##### Lane Risk Summary")
    st.dataframe(
        lane_agg.assign(
            Route=lambda d: d["Origin_City"] + " → " + d["Destination_City"],
            avg_score=lambda d: d["avg_score"].round(1),
            delay_rate=lambda d: (d["delay_rate"] * 100).round(1),
        )[["Route", "avg_score", "delay_rate", "volume"]]
        .rename(columns={"avg_score": "Avg Risk Score", "delay_rate": "Delay Rate (%)", "volume": "Volume"})
        .sort_values("Avg Risk Score", ascending=False),
        width='stretch', hide_index=True,
    )

# ============================== EXPLORER ===================================
with tab_explorer:
    st.markdown("#### Shipment-Level Explorer")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            fdf, x="Geopolitical_Risk_Index", y="Shipment_Risk_Score",
            color="Risk_Tier", color_discrete_map=TIER_COLORS,
            hover_data=["Origin_City", "Destination_City", "Transportation_Mode", "Product_Category"],
            opacity=0.6,
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827", height=420)
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig = px.pie(fdf, names="Risk_Tier", color="Risk_Tier", color_discrete_map=TIER_COLORS, hole=0.55)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111827", height=420,
                           legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, width='stretch')

    st.markdown("##### Highest-Risk Shipments")
    show_cols = [
        "Order_ID", "Route", "Transportation_Mode", "Product_Category",
        "Predicted_Delay_Probability", "Predicted_Delay_Days", "Shipment_Risk_Score", "Risk_Tier", "Is_Delayed",
    ]
    top = fdf.sort_values("Shipment_Risk_Score", ascending=False)[show_cols].head(200).copy()
    top["Predicted_Delay_Probability"] = (top["Predicted_Delay_Probability"] * 100).round(1)
    st.dataframe(
        top.rename(columns={"Predicted_Delay_Probability": "P(Delay) %", "Predicted_Delay_Days": "Expected Delay (days)"}),
        width='stretch', hide_index=True, height=400,
    )
    st.download_button(
        "⬇️ Download filtered scored shipments (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="filtered_scored_shipments.csv",
        mime="text/csv",
    )

# ============================== SCORER ======================================
with tab_scorer:
    st.markdown("#### Score a Hypothetical Shipment")
    st.caption("Simulate a shipment before it ships — the kind of check a trade-finance desk would run before underwriting it.")

    cities = sorted(CITY_COORDS.keys())
    with st.form("scorer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            origin = st.selectbox("Origin", cities, index=cities.index("Shenzhen, CN") if "Shenzhen, CN" in cities else 0)
            destination = st.selectbox("Destination", cities, index=cities.index("Rotterdam, NL") if "Rotterdam, NL" in cities else 1)
            route_type = st.selectbox("Route Type", sorted(df["Route_Type"].unique()))
        with c2:
            mode = st.selectbox("Transportation Mode", sorted(df["Transportation_Mode"].unique()))
            category = st.selectbox("Product Category", sorted(df["Product_Category"].unique()))
            order_month = st.selectbox("Order Month", list(range(1, 13)), index=5)
        with c3:
            base_lead = st.number_input("Base Lead Time (days)", min_value=1, max_value=60, value=28)
            sched_lead = st.number_input("Scheduled Lead Time (days)", min_value=1, max_value=70, value=30)
            day_of_week = st.selectbox("Order Day of Week (0=Mon)", list(range(7)), index=2)

        c4, c5, c6 = st.columns(3)
        with c4:
            geo_risk = st.slider("Geopolitical Risk Index", 0.0, 1.0, 0.5, 0.01)
        with c5:
            weather_risk = st.slider("Weather Severity Index", 0.0, 10.0, 5.0, 0.1)
        with c6:
            inflation = st.slider("Inflation Rate (%)", -1.0, 8.0, 3.5, 0.1)

        c7, c8 = st.columns(2)
        with c7:
            shipping_cost = st.number_input("Shipping Cost (USD)", min_value=1.0, value=9000.0, step=100.0)
        with c8:
            order_weight = st.number_input("Order Weight (Kg)", min_value=1.0, value=7000.0, step=50.0)

        submitted = st.form_submit_button("🎯 Compute Risk Score", width='stretch')

    if submitted:
        result = score_single_shipment({
            "base_lead_time": base_lead, "scheduled_lead_time": sched_lead,
            "geo_risk": geo_risk, "weather_risk": weather_risk, "inflation": inflation,
            "shipping_cost": shipping_cost, "order_weight": order_weight,
            "order_month": order_month, "day_of_week": day_of_week,
            "origin": origin, "destination": destination, "route_type": route_type,
            "mode": mode, "category": category,
        })

        st.markdown("---")
        colA, colB, colC = st.columns(3)
        colA.metric("Delay Probability", f"{result['probability']*100:.1f}%")
        colB.metric("Expected Delay", f"{result['expected_delay_days']:.1f} days")
        colC.metric("Shipment Risk Score", f"{result['risk_score']:.1f} / 100")

        badge_cls = BADGE_CLASS[result["risk_tier"]]
        st.markdown(
            f"<div style='margin-top:8px;'>Risk Tier: <span class='risk-badge {badge_cls}'>{result['risk_tier'].upper()}</span></div>",
            unsafe_allow_html=True,
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["risk_score"],
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": TIER_COLORS[result["risk_tier"]]},
                "steps": [
                    {"range": [0, 20], "color": "rgba(45,212,191,0.25)"},
                    {"range": [20, 40], "color": "rgba(245,185,66,0.25)"},
                    {"range": [40, 65], "color": "rgba(255,140,66,0.25)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,0.25)"},
                ],
            },
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111827", height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig, width='stretch')

        if result["risk_tier"] in ["High", "Severe"]:
            st.warning("Underwriting note: this shipment profile falls in the High/Severe risk band — consider a collateral haircut or margin adjustment before extending trade finance against it.")
        else:
            st.success("Underwriting note: this shipment profile falls in the Low/Medium risk band — standard terms are reasonable.")

# ============================== EXPLAINABILITY ===============================
with tab_explain:
    st.markdown("#### Why the Model Flags Risk the Way It Does")
    st.caption("SHAP feature importance from the delay-probability classifier, computed offline on the full training set.")

    import os
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    shap_bar = os.path.join(img_dir, "shap_top_drivers.png")
    shap_summary = os.path.join(img_dir, "shap_summary.png")

    if os.path.exists(shap_bar):
        st.image(shap_bar, width='stretch')
    if os.path.exists(shap_summary):
        st.image(shap_summary, width='stretch')

    st.markdown("---")
    st.markdown(
        """
        **Key takeaway:** cost-per-kg, base lead time, and schedule buffer dominate the model's decisions —
        far more than the continuous risk indices (Geopolitical Risk, Weather Severity). In this dataset,
        disruption is concentrated in specific lane × mode × category combinations rather than driven
        continuously by the risk indices. That's a deliberate, honest finding this dashboard is built around,
        not a limitation hidden from the viewer.
        """
    )

st.markdown("---")
st.markdown(
    "<span class='small-caption'>Supply Chain Disruption Risk Engine · Built with XGBoost, SHAP, and Streamlit · Leading-indicator model, no outcome leakage</span>",
    unsafe_allow_html=True,
)
