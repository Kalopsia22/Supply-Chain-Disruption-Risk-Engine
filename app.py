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
from dl_model import score_single_shipment_dl
from weather_api import fetch_port_conditions, classify_conditions

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

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
tab_overview, tab_map, tab_explorer, tab_scorer, tab_explain, tab_ailab, tab_live = st.tabs(
    ["📊 Portfolio Overview", "🌍 Global Risk Map", "🔍 Risk Explorer", "🎯 Shipment Risk Scorer", "🧠 Model Explainability", "🤖 Advanced AI Lab", "🛰️ Live Port Conditions"]
)

# ============================== OVERVIEW ==================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Shipments (filtered)", f"{len(fdf):,}")
    c2.metric("⏱️ Actual Delay Rate", f"{fdf['Is_Delayed'].mean()*100:.1f}%")
    c3.metric("🎯 Avg Risk Score", f"{fdf['Shipment_Risk_Score'].mean():.1f} / 100")
    c4.metric("🔥 High + Severe Tier", f"{(fdf['Risk_Tier'].isin(['High','Severe'])).mean()*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        col_donut, col_calib = st.columns([1, 1.4])
        with col_donut:
            st.markdown("##### 🍩 Risk Tier Split")
            tier_counts = fdf["Risk_Tier"].value_counts().reindex(["Low", "Medium", "High", "Severe"]).fillna(0)
            fig = go.Figure(go.Pie(
                labels=tier_counts.index, values=tier_counts.values, hole=0.58,
                marker=dict(colors=[TIER_COLORS[t] for t in tier_counts.index]),
                textinfo="label+percent", textfont=dict(color="#e5e7eb", size=12),
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#111827", height=330,
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                annotations=[dict(text=f"{len(fdf):,}<br>shipments", x=0.5, y=0.5,
                                   font_size=15, showarrow=False, font_color="#e5e7eb")],
            )
            st.plotly_chart(fig, width="stretch")
        with col_calib:
            st.markdown("##### ✅ Calibration Check — Score vs Reality")
            calib = fdf.groupby("Risk_Tier")["Is_Delayed"].mean().reindex(["Low", "Medium", "High", "Severe"])
            fig = go.Figure(go.Bar(
                x=calib.index, y=calib.values * 100,
                marker_color=[TIER_COLORS[t] for t in calib.index],
                text=[f"{v*100:.1f}%" for v in calib.values], textposition="outside",
            ))
            fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                               height=330, yaxis_title="Actual Delay Rate (%)", margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Each tier's actual delay rate should climb monotonically left to right — that's what confirms the score is well-calibrated, not just confident.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("##### 🌊 Shipment Flow: Mode → Route → Risk Outcome")
        st.caption("Sankey diagram — band thickness = shipment volume. Follow a band to see how shipments on a given mode/route tend to resolve.")

        sankey_df = fdf.groupby(["Transportation_Mode", "Route_Type", "Risk_Tier"]).size().reset_index(name="count")
        sankey_df = sankey_df[sankey_df["count"] > 0]

        modes_u = sorted(fdf["Transportation_Mode"].unique())
        routes_u = sorted(fdf["Route_Type"].unique())
        tiers_u = ["Low", "Medium", "High", "Severe"]

        node_labels = modes_u + routes_u + tiers_u
        node_colors = (
            ["#5b8def"] * len(modes_u)
            + ["#9b59b6"] * len(routes_u)
            + [TIER_COLORS[t] for t in tiers_u]
        )

        sources, targets, values, link_colors = [], [], [], []
        mode_route = fdf.groupby(["Transportation_Mode", "Route_Type"]).size().reset_index(name="count")
        for _, r in mode_route.iterrows():
            sources.append(modes_u.index(r["Transportation_Mode"]))
            targets.append(len(modes_u) + routes_u.index(r["Route_Type"]))
            values.append(r["count"])
            link_colors.append("rgba(91,141,239,0.35)")
        for _, r in sankey_df.iterrows():
            sources.append(len(modes_u) + routes_u.index(r["Route_Type"]))
            targets.append(len(modes_u) + len(routes_u) + tiers_u.index(r["Risk_Tier"]))
            values.append(r["count"])
            link_colors.append("rgba(155,89,182,0.35)")

        fig = go.Figure(go.Sankey(
            node=dict(label=node_labels, color=node_colors, pad=18, thickness=16,
                      line=dict(color="#0b0f19", width=0.5)),
            link=dict(source=sources, target=targets, value=values, color=link_colors),
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111827", height=420,
                           margin=dict(t=10, b=10, l=10, r=10),
                           font=dict(color="#e5e7eb", size=12))
        st.plotly_chart(fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    col_tree, col_mode = st.columns([1.3, 1])
    with col_tree:
        with st.container(border=True):
            st.markdown("##### 🗂️ Product Category Risk Map")
            st.caption("Box size = shipment volume · Color = delay rate")
            cat_agg = fdf.groupby("Product_Category").agg(
                volume=("Product_Category", "count"), delay_rate=("Is_Delayed", "mean"),
            ).reset_index()
            fig = px.treemap(
                cat_agg, path=["Product_Category"], values="volume",
                color="delay_rate", color_continuous_scale="RdYlGn_r",
                custom_data=["delay_rate"],
            )
            fig.update_traces(
                texttemplate="<b>%{label}</b><br>%{value} shipments<br>%{customdata[0]:.1%} delayed",
                textfont_size=13,
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#111827", height=380,
                               margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

    with col_mode:
        with st.container(border=True):
            st.markdown("##### 🚢 Delay Rate by Mode")
            rate_mode = fdf.groupby("Transportation_Mode")["Is_Delayed"].mean().sort_values(ascending=False)
            fig = px.bar(x=rate_mode.values, y=rate_mode.index, orientation="h",
                          color=rate_mode.values, color_continuous_scale="Teal")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                               height=380, showlegend=False, coloraxis_showscale=False,
                               xaxis_title="Delay Rate", yaxis_title="", margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("##### 📈 Delay Rate Trend Over Time")
        st.caption("Line = actual delay rate per month · Bars = shipment volume per month")
        trend = fdf.groupby("Order_YearMonth").agg(
            delay_rate=("Is_Delayed", "mean"), volume=("Is_Delayed", "count"),
        ).reset_index().sort_values("Order_YearMonth")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=trend["Order_YearMonth"], y=trend["volume"], name="Shipment Volume",
            marker_color="rgba(91,141,239,0.35)", yaxis="y2",
        ))
        fig.add_trace(go.Scatter(
            x=trend["Order_YearMonth"], y=trend["delay_rate"] * 100, name="Delay Rate (%)",
            mode="lines+markers", line=dict(color="#ef4444", width=2.5), marker=dict(size=6),
        ))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
            height=380, margin=dict(t=10, b=10),
            yaxis=dict(title="Delay Rate (%)"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Watch for months where delay rate spikes independent of volume — that points to a period effect (seasonality, geopolitical event) rather than simple congestion from more shipments.")

# ============================== MAP ========================================
with tab_map:
    st.markdown("#### 🌍 Trade Lane Risk Map")
    st.caption("Line color = average predicted risk score on that lane · Line width = shipment volume")
    st.markdown(
        "".join([f"<span class='risk-badge {BADGE_CLASS[t]}' style='margin-right:6px;'>{t}</span>" for t in ["Low", "Medium", "High", "Severe"]]),
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

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

    st.markdown("##### 🛣️ Lane Risk Summary")
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
    st.markdown("#### 🔍 Shipment-Level Explorer")
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown("##### Density: Geopolitical Risk vs Risk Score")
        st.caption("Heatmap of shipment concentration — avoids the overplotting a raw scatter of thousands of points would create.")
        fig = px.density_heatmap(
            fdf, x="Geopolitical_Risk_Index", y="Shipment_Risk_Score",
            nbinsx=25, nbinsy=25, color_continuous_scale="Teal",
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                           height=440, coloraxis_colorbar=dict(title="Shipments"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Notice there's no diagonal trend — risk score is high across the full range of geopolitical risk index, reinforcing that this feature isn't actually driving the score much (consistent with the SHAP findings).")
    with col2:
        st.markdown("##### Risk Score Spread by Tier")
        fig = go.Figure()
        for t in ["Low", "Medium", "High", "Severe"]:
            sub = fdf.loc[fdf["Risk_Tier"] == t, "Shipment_Risk_Score"]
            if len(sub) > 0:
                fig.add_trace(go.Violin(
                    y=sub, name=t, box_visible=True, meanline_visible=True,
                    line_color=TIER_COLORS[t], fillcolor=TIER_COLORS[t], opacity=0.55,
                ))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                           height=440, showlegend=False, yaxis_title="Shipment Risk Score")
        st.plotly_chart(fig, width="stretch")
        st.caption("Violin width = density of shipments at that score. Tight, non-overlapping shapes confirm clean tier separation.")

    st.markdown("##### 🔥 Highest-Risk Shipments")
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
    st.markdown("#### 🎯 Score a Hypothetical Shipment")
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

        st.markdown("---")
        st.markdown("##### 🧠 Deep Learning Cross-Check (PyTorch)")
        st.caption("A second, independently-trained model — a neural net with learned embeddings for route/mode/category — scoring the same shipment. Large disagreement between the two models is itself a useful signal.")

        dl_inputs = {
            "base_lead_time": base_lead, "scheduled_lead_time": sched_lead,
            "geo_risk": geo_risk, "weather_risk": weather_risk, "inflation": inflation,
            "shipping_cost": shipping_cost, "order_weight": order_weight,
            "order_month": order_month, "day_of_week": day_of_week,
            "origin": origin, "destination": destination, "route_type": route_type,
            "mode": mode, "category": category,
        }
        try:
            dl_result = score_single_shipment_dl(dl_inputs)
            dl_prob = dl_result["probability"]
            diff = abs(dl_prob - result["probability"])

            colD1, colD2, colD3 = st.columns(3)
            colD1.metric("XGBoost (Gradient Boosting)", f"{result['probability']*100:.1f}%")
            colD2.metric("Deep Neural Net (PyTorch)", f"{dl_prob*100:.1f}%")
            if diff < 0.15:
                colD3.metric("Agreement", "✅ Close")
            elif diff < 0.40:
                colD3.metric("Agreement", "⚠️ Some Disagreement")
            else:
                colD3.metric("Agreement", "🚨 High Disagreement")

            if diff >= 0.40:
                st.warning(
                    "These two independently-trained models disagree substantially on this shipment. "
                    "That can happen when an input sits in a region the training data doesn't pin down "
                    "clearly (e.g. an unusual combination of lead time and cost for that route), or simply "
                    "because tree-based and neural-network models generalize differently at the edges of "
                    "their training distribution. Either way, treat this specific prediction with more "
                    "caution than usual and consider a manual review rather than trusting either number alone."
                )
        except FileNotFoundError:
            st.info("Deep learning model files not found — run `python dl_model.py` to train and save the PyTorch model first.")

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
        st.markdown("##### 🧠 Why This Score? (SHAP Explanation)")
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
        st.markdown("##### 🚨 Anomaly Check (Isolation Forest)")
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
    st.markdown("#### 🧠 Why the Model Flags Risk the Way It Does")
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

    st.markdown("##### 🐝 Distribution of Impact (Beeswarm)")
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
    st.markdown("#### 🤖 Beyond the Core Model: Unsupervised & Simulation Techniques")
    st.caption("These sections use different ML techniques than the classifier/regressor above — unsupervised anomaly detection, unsupervised clustering, and Monte Carlo simulation — to surface patterns the supervised risk score alone doesn't show.")

    lab_anomaly, lab_cluster, lab_montecarlo = st.tabs(
        ["🚨 Anomaly Detection", "🧬 Risk Archetypes (Clustering)", "🎲 Portfolio Monte Carlo Simulation"]
    )

    # --- Anomaly Detection sub-tab ---
    with lab_anomaly:
        st.markdown("##### 🌲 Isolation Forest — Unusual Shipment Detection")
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

        st.markdown("###### Where Anomalies Sit vs the Normal Population")
        st.caption("Background contour = density of normal shipments (no clutter from thousands of points) · Red markers = the flagged anomalies only")

        normal = adf_f[~adf_f["Is_Anomaly"]]
        anomalous = adf_f[adf_f["Is_Anomaly"]]

        fig = go.Figure()
        fig.add_trace(go.Histogram2dContour(
            x=normal["Cost_Per_Kg"], y=normal["Shipment_Risk_Score"],
            colorscale="Teal", showscale=False, opacity=0.85,
            contours=dict(coloring="fill"),
        ))
        fig.add_trace(go.Scatter(
            x=anomalous["Cost_Per_Kg"], y=anomalous["Shipment_Risk_Score"],
            mode="markers", marker=dict(color="#ef4444", size=6, line=dict(width=1, color="#0b0f19")),
            name="Anomaly",
            hovertext=[f"{o} → {d}<br>{m}, {c}" for o, d, m, c in zip(
                anomalous["Origin_City"], anomalous["Destination_City"],
                anomalous["Transportation_Mode"], anomalous["Product_Category"])],
            hoverinfo="text",
        ))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827", height=440,
            xaxis_title="Cost Per Kg", yaxis_title="Shipment Risk Score", showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("###### 🚩 Top Flagged Shipments")
        show_cols = ["Order_ID", "Route", "Transportation_Mode", "Product_Category", "Cost_Per_Kg", "Shipment_Risk_Score", "Risk_Tier", "Anomaly_Score"]
        st.dataframe(
            adf_f[adf_f["Is_Anomaly"]].sort_values("Anomaly_Score", ascending=False)[show_cols].head(50),
            width="stretch", hide_index=True, height=320,
        )

    # --- Clustering sub-tab ---
    with lab_cluster:
        st.markdown("##### 🧬 K-Means Risk Archetypes")
        st.caption("Unsupervised segmentation on cost, weight, lead time, and risk-index features (not on the delay label). Reveals natural portfolio groupings a trade-finance desk could price or monitor differently.")

        n_clusters = st.slider("Number of archetypes", 3, 8, 5)

        @st.cache_data(show_spinner="Running K-Means + PCA...")
        def _clusters(k):
            return compute_clusters(df, n_clusters=k)

        cdf, profile = _clusters(n_clusters)
        cdf_f = cdf.loc[cdf.index.isin(fdf.index)]

        st.markdown("###### 📋 Archetype Profiles")
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

        st.markdown("###### 🗺️ Cluster Map (PCA Projection)")
        st.caption("Contour outlines instead of raw points — each line shows where one archetype's shipments concentrate in the 2D projection, without thousands of overlapping dots.")
        fig = go.Figure()
        palette = px.colors.qualitative.Set2
        for i, c in enumerate(sorted(cdf_f["Cluster"].unique())):
            sub = cdf_f[cdf_f["Cluster"] == c]
            if len(sub) < 5:
                continue
            fig.add_trace(go.Histogram2dContour(
                x=sub["PCA_1"], y=sub["PCA_2"], name=f"Cluster {c}",
                showscale=False, ncontours=6,
                contours=dict(coloring="lines", showlabels=False),
                line=dict(width=2, color=palette[i % len(palette)]),
            ))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827", height=460,
                           legend_title_text="Cluster", showlegend=True,
                           xaxis_title="PCA Component 1", yaxis_title="PCA Component 2")
        st.plotly_chart(fig, width="stretch")
        st.caption("Projected from 7 numeric features down to 2D via PCA. Separated contour shapes mean the underlying feature combinations are genuinely different, not just relabeled risk tiers.")

        st.markdown("###### 🕸️ Archetype Comparison (Radar)")
        st.caption("Each axis normalized 0-1 across archetypes, so shapes are directly comparable regardless of original units.")

        radar_dims = ["avg_cost_per_kg", "avg_risk_score", "delay_rate"]
        radar_extra = cdf.groupby("Cluster").agg(
            avg_base_lead=("Base_Lead_Time_Days", "mean"),
            avg_geo_risk=("Geopolitical_Risk_Index", "mean"),
        ).reset_index()
        radar_profile = profile.merge(radar_extra, on="Cluster")

        radar_cols = {
            "avg_cost_per_kg": "Cost/kg",
            "avg_base_lead": "Lead Time",
            "avg_geo_risk": "Geo Risk",
            "delay_rate": "Delay Rate",
            "avg_risk_score": "Risk Score",
        }
        norm_df = radar_profile.copy()
        for col in radar_cols:
            rng = norm_df[col].max() - norm_df[col].min()
            norm_df[col] = (norm_df[col] - norm_df[col].min()) / rng if rng > 0 else 0.5

        fig = go.Figure()
        palette = px.colors.qualitative.Set2
        for i, row in norm_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[c] for c in radar_cols] + [row[list(radar_cols)[0]]],
                theta=list(radar_cols.values()) + [list(radar_cols.values())[0]],
                fill="toself", name=f"Cluster {int(row['Cluster'])}",
                line=dict(color=palette[i % len(palette)]),
                opacity=0.5,
            ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111827", height=460,
            polar=dict(bgcolor="#111827", radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    # --- Monte Carlo sub-tab ---
    with lab_montecarlo:
        st.markdown("##### 🎲 Portfolio Exposure Simulation")
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

# ============================== LIVE PORT CONDITIONS ===============================
with tab_live:
    st.markdown("#### 🛰️ Live Port Conditions")
    st.caption("Real marine weather from Open-Meteo (free, no API key) — current wave height, wind, and a 7-day hourly forecast per port. Updates on a live basis, not from the static dataset.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.4, 1, 1])
    with col_ctrl1:
        refresh_minutes = st.select_slider("Auto-refresh every", options=[1, 5, 10, 15, 30], value=10, format_func=lambda m: f"{m} min")
    with col_ctrl2:
        auto_on = st.toggle("Auto-refresh", value=False)
    with col_ctrl3:
        manual_refresh = st.button("🔄 Refresh Now")

    if auto_on:
        if AUTOREFRESH_AVAILABLE:
            st_autorefresh(interval=refresh_minutes * 60 * 1000, key="live_conditions_refresh")
        else:
            st.warning("Install `streamlit-autorefresh` (already in requirements.txt) to enable automatic refresh; using manual refresh for now.")

    @st.cache_data(ttl=600, show_spinner=False)
    def _cached_port_conditions(port_name, lat, lon, _cache_bust):
        return fetch_port_conditions(lat, lon)

    if manual_refresh:
        _cached_port_conditions.clear()

    import datetime
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    fetch_time_bucket = now_utc.strftime("%Y%m%d%H%M") if manual_refresh else now_utc.strftime("%Y%m%d%H")

    st.markdown(f"<span class='small-caption'>Last fetched (UTC, rounded to cache window): {now_utc.strftime('%Y-%m-%d %H:%M')}</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected_port = st.selectbox("Select a port", sorted(CITY_COORDS.keys()))
    lat, lon = CITY_COORDS[selected_port]

    with st.spinner(f"Fetching live conditions for {selected_port}..."):
        conditions = _cached_port_conditions(selected_port, lat, lon, fetch_time_bucket)

    if not conditions.get("ok"):
        st.error(
            f"Could not fetch live data for {selected_port}: {conditions.get('error', 'unknown error')}. "
            "If this persists, check that your Streamlit Cloud deployment has outbound internet access to "
            "api.open-meteo.com and marine-api.open-meteo.com."
        )
    else:
        classification = classify_conditions(conditions["current_wave_height_m"], conditions["current_wind_kmh"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌊 Current Wave Height", f"{conditions['current_wave_height_m']:.1f} m" if conditions["current_wave_height_m"] is not None else "n/a")
        c2.metric("💨 Current Wind Speed", f"{conditions['current_wind_kmh']:.0f} km/h" if conditions["current_wind_kmh"] is not None else "n/a")
        c3.metric("📈 Peak Wave (next 72h)", f"{conditions['peak_wave_72h_m']:.1f} m" if conditions["peak_wave_72h_m"] is not None else "n/a")
        c4.metric("📈 Peak Wind (next 72h)", f"{conditions['peak_wind_72h_kmh']:.0f} km/h" if conditions["peak_wind_72h_kmh"] is not None else "n/a")

        st.markdown(
            f"<span class='risk-badge' style='background: {classification['color']}22; color: {classification['color']}; border: 1px solid {classification['color']};'>{classification['level']}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Classification uses standard maritime wave-height/wind operational thresholds (Douglas sea state / Beaufort scale bands), not a trained model — this is a transparent rule, not a black box.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 7-Day Hourly Forecast")
        fc = conditions["df"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["time"], y=fc["wave_height_m"], name="Wave Height (m)",
                                   line=dict(color="#2dd4bf", width=2)))
        fig.add_trace(go.Scatter(x=fc["time"], y=fc["wind_speed_kmh"], name="Wind Speed (km/h)",
                                   line=dict(color="#f5b942", width=2), yaxis="y2"))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827", height=400,
            yaxis=dict(title="Wave Height (m)"),
            yaxis2=dict(title="Wind Speed (km/h)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.1), margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")
        st.markdown("##### 🔗 Apply to Shipment Scorer")
        st.caption(f"If you're scoring a shipment through {selected_port} right now, you can carry these live readings over as the Weather Severity Index input in the Scorer tab.")
        implied_severity = min(10.0, round((conditions["current_wave_height_m"] or 0) * 1.8 + (conditions["current_wind_kmh"] or 0) / 15, 1))
        st.info(f"Suggested Weather Severity Index based on live conditions at {selected_port}: **{implied_severity} / 10**")

st.markdown("---")
st.markdown(
    "<span class='small-caption'>Supply Chain Disruption Risk Engine · Built with XGBoost, PyTorch, SHAP, and Streamlit · Leading-indicator model, no outcome leakage</span>",
    unsafe_allow_html=True,
)
