"""
Supply Chain Disruption Risk Engine — Interactive Dashboard
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_utils import (
    load_scored_data, score_single_shipment, CITY_COORDS,
    compute_global_shap, compute_single_shap, clean_feature_name,
    compute_anomaly_scores, score_single_anomaly, compute_clusters,
    monte_carlo_portfolio, generate_underwriting_memo,
)

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
tab_overview, tab_map, tab_explorer, tab_scorer, tab_explain, tab_ailab = st.tabs(
    ["📊 Portfolio Overview", "🌍 Global Risk Map", "🔍 Risk Explorer", "🎯 Shipment Risk Scorer", "🧠 Model Explainability", "🤖 Advanced AI Lab"]
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

        st.markdown("---")
        st.markdown("##### Why this score? (SHAP explanation for this shipment)")
        with st.spinner("Computing SHAP explanation..."):
            sv_row, feat_row, base_value = compute_single_shap({
                "base_lead_time": base_lead, "scheduled_lead_time": sched_lead,
                "geo_risk": geo_risk, "weather_risk": weather_risk, "inflation": inflation,
                "shipping_cost": shipping_cost, "order_weight": order_weight,
                "order_month": order_month, "day_of_week": day_of_week,
                "origin": origin, "destination": destination, "route_type": route_type,
                "mode": mode, "category": category,
            })

        exp_df = pd.DataFrame({
            "feature": [clean_feature_name(f) for f in feat_row.index],
            "shap_value": sv_row,
        })
        exp_df["abs"] = exp_df["shap_value"].abs()
        exp_df = exp_df.sort_values("abs", ascending=False).head(10).sort_values("shap_value")

        fig = go.Figure(go.Bar(
            x=exp_df["shap_value"], y=exp_df["feature"], orientation="h",
            marker_color=["#ef4444" if v > 0 else "#2dd4bf" for v in exp_df["shap_value"]],
            text=[f"{v:+.3f}" for v in exp_df["shap_value"]], textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
            height=380, margin=dict(t=20, b=10, l=10, r=50),
            xaxis_title="SHAP value (red = pushes risk up, teal = pushes risk down)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Base rate (average predicted risk before considering this shipment's features): {base_value:.3f}. Bars show how each feature moved this specific shipment away from that baseline.")

        st.markdown("---")
        st.markdown("##### Anomaly Check (Isolation Forest)")
        st.caption("Independent of the risk score — flags shipments whose feature combination is statistically unusual (e.g. an oddly cheap or oddly heavy shipment for its category), which can matter for quality/fraud review even on shipments the risk model considers low-risk.")

        @st.cache_resource(show_spinner="Fitting anomaly detector...")
        def _get_anomaly_model():
            _, model, scaler = compute_anomaly_scores(df)
            return model, scaler

        anomaly_model, anomaly_scaler = _get_anomaly_model()
        anomaly_result = score_single_anomaly({
            "base_lead_time": base_lead, "scheduled_lead_time": sched_lead,
            "geo_risk": geo_risk, "weather_risk": weather_risk, "inflation": inflation,
            "shipping_cost": shipping_cost, "order_weight": order_weight,
            "order_month": order_month, "day_of_week": day_of_week,
            "origin": origin, "destination": destination, "route_type": route_type,
            "mode": mode, "category": category,
        }, anomaly_model, anomaly_scaler)

        if anomaly_result["is_anomaly"]:
            st.error(f"⚠️ Flagged as a statistical outlier (anomaly score: {anomaly_result['anomaly_score']:.3f}). This shipment's cost/weight/lead-time profile doesn't resemble typical shipments in the historical data — worth a manual look regardless of its risk tier.")
        else:
            st.info(f"✅ Not flagged as an outlier (anomaly score: {anomaly_result['anomaly_score']:.3f}). This shipment's profile is consistent with typical historical shipments.")

        st.markdown("---")
        st.markdown("##### 📝 Auto-Generated Underwriting Memo")
        shipment_desc = f"{origin} → {destination}, {mode}, {category}"
        memo = generate_underwriting_memo(shipment_desc, result, sv_row, feat_row)
        st.markdown(f"> {memo}")

# ============================== EXPLAINABILITY ===============================
with tab_explain:
    st.markdown("#### Why the Model Flags Risk the Way It Does")
    st.caption("SHAP (SHapley Additive exPlanations) values computed live from the trained classifier — not a static image, so this always matches the model actually running.")

    @st.cache_data(show_spinner="Computing SHAP values...")
    def _global_shap():
        shap_values, X_trans_df = compute_global_shap(df, sample_size=800)
        return shap_values, X_trans_df

    shap_values, X_trans_df = _global_shap()

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    labels = [clean_feature_name(X_trans_df.columns[i]) for i in order][::-1]
    values = [mean_abs[i] for i in order][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#2dd4bf",
        text=[f"{v:.3f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
        height=480, margin=dict(t=20, b=10, l=10, r=40),
        xaxis_title="Mean |SHAP value| — average impact on predicted delay risk",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Distribution of Impact (Beeswarm)")
    st.caption("Each dot is one shipment. Position = SHAP value (pushes risk up or down). Color = whether that feature's value was high (red) or low (blue) for that shipment.")

    top_n = 12
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    beeswarm_fig = go.Figure()
    for rank, i in enumerate(top_idx[::-1]):
        col = X_trans_df.columns[i]
        vals = X_trans_df[col].values
        sv = shap_values[:, i]
        norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        jitter = (np.random.rand(len(sv)) - 0.5) * 0.6
        beeswarm_fig.add_trace(go.Scatter(
            x=sv, y=[rank + j for j in jitter],
            mode="markers",
            marker=dict(size=5, color=norm, colorscale="RdBu_r", opacity=0.6,
                        line=dict(width=0)),
            name=clean_feature_name(col),
            showlegend=False,
            hovertext=[f"{clean_feature_name(col)}: {v:.2f}" for v in vals],
            hoverinfo="text",
        ))
    beeswarm_fig.update_layout(
        template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
        height=460, margin=dict(t=20, b=10, l=10, r=10),
        yaxis=dict(tickmode="array", tickvals=list(range(top_n)),
                   ticktext=[clean_feature_name(X_trans_df.columns[i]) for i in top_idx[::-1]]),
        xaxis_title="SHAP value (impact on predicted delay risk)",
    )
    st.plotly_chart(beeswarm_fig, width="stretch")

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

# ============================== ADVANCED AI LAB ===============================
with tab_ailab:
    st.markdown("#### Beyond the Core Model: Unsupervised & Simulation Techniques")
    st.caption("These sections use different ML techniques than the classifier/regressor above — unsupervised anomaly detection, unsupervised clustering, and Monte Carlo simulation — to surface patterns the supervised risk score alone doesn't show.")

    lab_anomaly, lab_cluster, lab_montecarlo = st.tabs(
        ["🚨 Anomaly Detection", "🧬 Risk Archetypes (Clustering)", "🎲 Portfolio Monte Carlo Simulation"]
    )

    # --- Anomaly Detection sub-tab ---
    with lab_anomaly:
        st.markdown("##### Isolation Forest — Unusual Shipment Detection")
        st.caption("Unsupervised — doesn't use the delay label at all. Flags shipments whose cost/weight/lead-time/risk-index combination is statistically unusual relative to the rest of the portfolio. Deliberately independent of the risk score: a shipment can be Low risk-tier and still get flagged here.")

        @st.cache_data(show_spinner="Fitting Isolation Forest...")
        def _anomaly_df():
            adf, _, _ = compute_anomaly_scores(df)
            return adf

        adf = _anomaly_df()
        adf_f = adf.loc[fdf.index]

        c1, c2, c3 = st.columns(3)
        c1.metric("Anomalies Flagged", f"{adf_f['Is_Anomaly'].sum():,}")
        c2.metric("% of Filtered Portfolio", f"{adf_f['Is_Anomaly'].mean()*100:.1f}%")
        c3.metric("Anomaly + Low Risk Tier", f"{((adf_f['Is_Anomaly']) & (adf_f['Risk_Tier']=='Low')).sum():,}")
        st.caption("The third metric is the interesting one — shipments the risk model considers safe, but that are structurally unusual. Worth a quality/fraud review pass independent of delay risk.")

        fig = px.scatter(
            adf_f, x="Cost_Per_Kg", y="Shipment_Risk_Score",
            color="Is_Anomaly", color_discrete_map={True: "#ef4444", False: "#2dd4bf"},
            hover_data=["Origin_City", "Destination_City", "Transportation_Mode", "Product_Category"],
            opacity=0.6, labels={"Is_Anomaly": "Anomaly"},
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827", height=440)
        st.plotly_chart(fig, width="stretch")

        st.markdown("###### Top Flagged Shipments")
        show_cols = ["Order_ID", "Route", "Transportation_Mode", "Product_Category", "Cost_Per_Kg", "Shipment_Risk_Score", "Risk_Tier", "Anomaly_Score"]
        st.dataframe(
            adf_f[adf_f["Is_Anomaly"]].sort_values("Anomaly_Score", ascending=False)[show_cols].head(50),
            width="stretch", hide_index=True, height=320,
        )

    # --- Clustering sub-tab ---
    with lab_cluster:
        st.markdown("##### K-Means Risk Archetypes")
        st.caption("Unsupervised segmentation on cost, weight, lead time, and risk-index features (not on the delay label). Reveals natural portfolio groupings a trade-finance desk could price or monitor differently.")

        n_clusters = st.slider("Number of archetypes", 3, 8, 5)

        @st.cache_data(show_spinner="Running K-Means + PCA...")
        def _clusters(k):
            return compute_clusters(df, n_clusters=k)

        cdf, profile = _clusters(n_clusters)
        cdf_f = cdf.loc[cdf.index.isin(fdf.index)]

        st.markdown("###### Archetype Profiles")
        display_profile = profile.copy()
        display_profile["avg_risk_score"] = display_profile["avg_risk_score"].round(1)
        display_profile["delay_rate"] = (display_profile["delay_rate"] * 100).round(1)
        display_profile["avg_cost"] = display_profile["avg_cost"].round(0)
        display_profile["avg_cost_per_kg"] = display_profile["avg_cost_per_kg"].round(2)
        st.dataframe(
            display_profile[["Cluster", "archetype_label", "size", "avg_risk_score", "delay_rate", "avg_cost_per_kg"]]
            .rename(columns={"avg_risk_score": "Avg Risk Score", "delay_rate": "Delay Rate (%)", "avg_cost_per_kg": "Avg Cost/Kg", "size": "Shipments"}),
            width="stretch", hide_index=True,
        )

        st.markdown("###### Cluster Map (PCA Projection)")
        fig = px.scatter(
            cdf_f, x="PCA_1", y="PCA_2", color=cdf_f["Cluster"].astype(str),
            hover_data=["Transportation_Mode", "Product_Category", "Shipment_Risk_Score"],
            opacity=0.6, color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827", height=460,
                           legend_title_text="Cluster")
        st.plotly_chart(fig, width="stretch")
        st.caption("Each point is one shipment, projected from 7 numeric features down to 2D via PCA. Distinct clusters here mean the underlying feature combinations are genuinely different, not just relabeled risk tiers.")

    # --- Monte Carlo sub-tab ---
    with lab_montecarlo:
        st.markdown("##### Portfolio Exposure Simulation")
        st.caption("Treats each shipment's predicted delay probability as an independent Bernoulli trial and simulates the filtered portfolio thousands of times — a lightweight, VaR-style view of tail risk, not just the average case.")

        n_sims = st.select_slider("Number of simulations", options=[500, 1000, 2000, 5000], value=2000)

        if len(fdf) == 0:
            st.warning("No shipments match the current filters.")
        else:
            with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
                mc = monte_carlo_portfolio(fdf, n_simulations=n_sims)

            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Delayed Shipments", f"{mc['expected_delayed']:.0f}")
            c2.metric("95th Pctile Delayed (tail case)", f"{mc['p95_delayed']:.0f}")
            c3.metric("Expected Financial Exposure", f"${mc['expected_exposure']:,.0f}")

            c4, c5 = st.columns(2)
            c4.metric("95th Pctile Exposure", f"${mc['p95_exposure']:,.0f}")
            c5.metric("99th Pctile Exposure", f"${mc['p99_exposure']:,.0f}")

            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure(go.Histogram(x=mc["delayed_counts"], marker_color="#2dd4bf", nbinsx=40))
                fig.add_vline(x=mc["expected_delayed"], line_dash="dash", line_color="#f5b942",
                              annotation_text="Expected", annotation_font_color="#f5b942")
                fig.add_vline(x=mc["p95_delayed"], line_dash="dash", line_color="#ef4444",
                              annotation_text="95th pctile", annotation_font_color="#ef4444")
                fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                                   height=380, title="Distribution: Number of Delayed Shipments",
                                   xaxis_title="Delayed shipment count")
                st.plotly_chart(fig, width="stretch")
            with col_b:
                fig = go.Figure(go.Histogram(x=mc["exposure"], marker_color="#f5b942", nbinsx=40))
                fig.add_vline(x=mc["expected_exposure"], line_dash="dash", line_color="#2dd4bf",
                              annotation_text="Expected", annotation_font_color="#2dd4bf")
                fig.add_vline(x=mc["p95_exposure"], line_dash="dash", line_color="#ef4444",
                              annotation_text="95th pctile", annotation_font_color="#ef4444")
                fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                                   height=380, title="Distribution: Financial Exposure to Delay ($)",
                                   xaxis_title="Total exposure (USD)")
                st.plotly_chart(fig, width="stretch")

            st.markdown(
                f"""
                **Reading this:** on average, this filtered portfolio of **{len(fdf):,} shipments** is expected to see
                **{mc['expected_delayed']:.0f} delayed shipments**, representing roughly **${mc['expected_exposure']:,.0f}**
                in shipping cost exposed to delay. In a bad-case scenario (95th percentile across simulations), that could
                rise to **{mc['p95_delayed']:.0f} delayed shipments** and **${mc['p95_exposure']:,.0f}** of exposure —
                the kind of tail-risk figure a portfolio risk committee would want alongside the average.
                """
            )

st.markdown("---")
st.markdown(
    "<span class='small-caption'>Supply Chain Disruption Risk Engine · Built with XGBoost, SHAP, and Streamlit · Leading-indicator model, no outcome leakage</span>",
    unsafe_allow_html=True,
)
