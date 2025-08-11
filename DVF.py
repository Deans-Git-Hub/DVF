# dvf_app.py
"""
Synthetic Persona DVF Scoring Assistant — Demo Mode
---------------------------------------------------
Instant AI-like feedback on new concepts using hardcoded synthetic personas.
Run:
    pip install streamlit pandas numpy plotly
    streamlit run dvf_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="DVF Scoring Assistant", layout="wide")
st.title("📊 Synthetic Persona DVF Scoring Assistant")
st.caption("Instant AI-like feedback on new concepts (demo, hard-coded data)")
st.info("🚀 DEMO MODE: All controls visible, calculations local-only")

# ------------------------------
# Hardcoded demo data
# ------------------------------

PERSONAS = [
    {"id": "p01", "name": "Eco-Savvy Parent",     "prior": {"des": 3.8, "via": 3.1, "fea": 2.9}},
    {"id": "p02", "name": "Budget Millennial",    "prior": {"des": 3.2, "via": 2.8, "fea": 3.5}},
    {"id": "p03", "name": "Tech Enthusiast",      "prior": {"des": 4.5, "via": 4.0, "fea": 3.8}},
    {"id": "p04", "name": "Busy Executive",       "prior": {"des": 3.6, "via": 4.2, "fea": 3.0}},
    {"id": "p05", "name": "Eco-Advocate Student", "prior": {"des": 4.0, "via": 2.5, "fea": 3.2}},
    {"id": "p06", "name": "Health-Pro Professional","prior":{"des":3.9,"via":3.3,"fea":3.1}},
    {"id": "p07", "name": "Value-Seeker Retiree", "prior": {"des": 3.1, "via": 3.0, "fea": 2.8}},
    {"id": "p08", "name": "Adventure Traveler",   "prior": {"des": 4.2, "via": 3.7, "fea": 3.6}},
]

QUESTIONS = [
    {"id": "q1",  "dim": "des", "text": "How appealing is this concept?"},
    {"id": "q2",  "dim": "via", "text": "Likelihood to pay a premium?"},
    {"id": "q3",  "dim": "fea", "text": "Perceived technical feasibility?"},
    {"id": "q4",  "dim": "des", "text": "Would you recommend it to a friend?"},
    {"id": "q5",  "dim": "via", "text": "Would you consider subscription?"},
    {"id": "q6",  "dim": "fea", "text": "Integration ease with current tools?"},
    {"id": "q7",  "dim": "des", "text": "Emotional resonance of brand?"},
    {"id": "q8",  "dim": "via", "text": "Expected ROI over 1 year?"},
    {"id": "q9",  "dim": "fea", "text": "Support & maintenance adequacy?"},
    {"id": "q10", "dim": "des", "text": "Overall desirability rating?"},
]

# Pre-sampled responses per persona (1–5 scale)
RESPONSES = [
    {"persona": "p01", **{f"q{i}": v for i,v in enumerate([4.2,3.5,2.9,4.0,3.1,3.0,4.1,2.8,3.2,4.0], start=1)}},
    {"persona": "p02", **{f"q{i}": v for i,v in enumerate([3.0,2.6,3.7,3.2,2.9,3.5,3.1,2.7,3.0,3.2], start=1)}},
    {"persona": "p03", **{f"q{i}": v for i,v in enumerate([4.6,4.1,3.9,4.5,4.0,3.8,4.7,3.9,4.0,4.5], start=1)}},
    {"persona": "p04", **{f"q{i}": v for i,v in enumerate([3.7,4.3,3.2,3.8,4.1,3.1,3.6,4.2,3.0,3.6], start=1)}},
    {"persona": "p05", **{f"q{i}": v for i,v in enumerate([4.1,2.4,3.3,4.0,2.6,3.2,4.2,2.5,3.1,4.0], start=1)}},
    {"persona": "p06", **{f"q{i}": v for i,v in enumerate([3.9,3.2,3.0,3.7,3.3,3.1,3.8,3.4,3.2,3.9], start=1)}},
    {"persona": "p07", **{f"q{i}": v for i,v in enumerate([3.2,3.1,2.7,3.0,3.0,2.8,3.3,3.2,2.9,3.1], start=1)}},
    {"persona": "p08", **{f"q{i}": v for i,v in enumerate([4.3,3.8,3.6,4.1,3.7,3.6,4.4,3.9,3.5,4.2], start=1)}},
]

# Optional pre-laid coordinates for bubble placement
COORDINATES = [
    {"x": 3.1, "y": 4.5}, {"x": 2.8, "y": 3.2}, {"x": 4.5, "y": 4.2}, {"x": 3.9, "y": 3.6},
    {"x": 4.0, "y": 4.0}, {"x": 3.3, "y": 3.9}, {"x": 3.0, "y": 3.1}, {"x": 4.2, "y": 4.3"},
]

# ------------------------------
# Scoring helpers
# ------------------------------

def compute_dvf(responses: dict, prior: dict, w_prior: float):
    dims = {"des": [], "via": [], "fea": []}
    for q, v in responses.items():
        dim = next(item["dim"] for item in QUESTIONS if item["id"] == q)
        dims[dim].append(v)
    mean = {d: np.mean(vs) for d, vs in dims.items()}
    return {d: w_prior * prior[d] + (1 - w_prior) * mean[d] for d in mean}

def add_variance(score: float, sigma: float):
    return float(np.clip(np.random.normal(score, sigma), 1.0, 5.0))

# ------------------------------
# Sidebar controls
# ------------------------------
with st.sidebar:
    st.header("🔧 Controls")
    show_legend  = st.checkbox("Show feasibility legend", value=False)
    selected_ids = st.multiselect(
        "Select Personas",
        [p["name"] for p in PERSONAS],
        default=[p["name"] for p in PERSONAS]
    )
    sigma = st.slider(
        "Variance σ",
        min_value=0.05, max_value=0.50,
        step=0.05, value=0.15
    )
    w_prior = st.select_slider(
        "Prior weight",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        value=0.25
    )
    st.divider()
    st.subheader("🚧 Demo Filters (disabled)")
    st.file_uploader("Upload personas/answers", type=["json"], disabled=True)

# Filter personas/responses
mask      = [p["name"] in selected_ids for p in PERSONAS]
personas  = [p for p, m in zip(PERSONAS, mask) if m]
responses = [r for r in RESPONSES if r["persona"] in {p["id"] for p in personas}]
coords    = [c for c, m in zip(COORDINATES, mask) if m]

# ------------------------------
# Tabs
# ------------------------------
tab_map, tab_det, tab_var, tab_sum, tab_cmp, tab_exp = st.tabs([
    "DVF Map", "Persona Details", "Variance Matrix",
    "Summary", "Compare", "Export"
])

# ------------------------------
# DVF Map
# ------------------------------
with tab_map:
    st.subheader("DVF Bubble Map")
    rows = []
    for p, resp, coord in zip(personas, responses, coords):
        base = compute_dvf(
            {k: v for k, v in resp.items() if k != "persona"},
            p["prior"],
            w_prior
        )
        dvf = {d: add_variance(s, sigma) for d, s in base.items()}
        rows.append({**{"id": p["id"], "name": p["name"], **dvf}, **coord})

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x="via", y="des", size="fea", color="fea",
        color_continuous_scale="RdYlGn_r", range_color=[1, 5],
        size_max=60, opacity=0.9,
        hover_data={"name": True, "des":":.2f", "via":":.2f", "fea":":.2f"}
    )
    fig.update_traces(
        mode="markers+text",
        text=df["name"].str.split().str[0],
        textposition="middle center",
        marker=dict(line=dict(width=1, color="rgba(0,0,0,0.3)"))
    )
    if not show_legend:
        fig.update_coloraxes(showscale=False)
    fig.update_layout(
        xaxis_title="Viability",
        yaxis_title="Desirability",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        margin=dict(l=120, r=30, t=30, b=100)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------------
# Persona Details (stub)
# ------------------------------
with tab_det:
    st.subheader("Persona Details")
    st.info("🚧 Demo: Select a persona below to view DVF breakdown")
    sel = st.selectbox("Choose Persona", [p["name"] for p in personas])
    # TODO: render a card or heatmap for that persona's individual responses

# ------------------------------
# Variance Matrix (stub)
# ------------------------------
with tab_var:
    st.subheader("Question Variance Matrix")
    st.caption("Heatmap of how responses vary across personas")
    # TODO: build and plot a Plotly heatmap of each persona's answers per question

# ------------------------------
# Summary (stub)
# ------------------------------
with tab_sum:
    st.subheader("Summary")
    st.info("🚧 Demo: Metrics & bar charts go here")
    # TODO: compute avg DVF metrics, top/bottom personas, most divisive question

# ------------------------------
# Compare & Export (placeholders)
# ------------------------------
with tab_cmp:
    st.subheader("Compare Concepts")
    st.info("🚧 Demo: Upload another concept to compare panels")
    st.file_uploader("Upload comparison JSON", type=["json"], disabled=True)

with tab_exp:
    st.subheader("Export")
    st.info("🚧 Demo: Download CSV/JSON/PNG")
    c1, c2, c3 = st.columns(3)
    with c1: st.button("⬇️ CSV", disabled=True)
    with c2: st.button("⬇️ JSON", disabled=True)
    with c3: st.button("⬇️ PNG", disabled=True)
