# dvf_app.py
"""
Graza Dressing DVF Scoring Assistant — Demo Mode
------------------------------------------------
Instant synthetic-persona feedback on whether Graza should launch ready-to-use
olive-oil salad dressings.

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
st.set_page_config(page_title="Graza Dressing DVF Assistant", layout="wide")
st.title("🥗 Graza Dressing DVF Scoring Assistant")
st.caption("Synthetic-persona panel evaluating a potential Graza salad-dressing line")
st.info("🚀 DEMO MODE: All data hard-coded for instant play")

# ------------------------------
# 1 · Personas (research-backed synthetic)
# ------------------------------
PERSONAS = [
    {"id": "p01", "name": "Social-Chef Sara",       "prior": {"des": 4.5, "via": 3.8, "fea": 3.6}},
    {"id": "p02", "name": "Frugal-Dad Dan",         "prior": {"des": 3.2, "via": 2.6, "fea": 3.3}},
    {"id": "p03", "name": "Clean-Eating Chloe",     "prior": {"des": 4.3, "via": 3.9, "fea": 3.7}},
    {"id": "p04", "name": "Eco-Mom Maya",           "prior": {"des": 4.1, "via": 3.4, "fea": 3.2}},
    {"id": "p05", "name": "Chef-Next-Door Marco",   "prior": {"des": 3.0, "via": 3.2, "fea": 4.0}},
    {"id": "p06", "name": "Time-Starved Tina",      "prior": {"des": 4.2, "via": 3.6, "fea": 3.5}},
]

# ------------------------------
# 2 · DVF Questionnaire (dressings-specific)
# ------------------------------
QUESTIONS = [
    {"id": "q1", "dim": "des",
     "text": "How appealing is a Graza line of ready-to-use olive-oil salad dressings?"},
    {"id": "q2", "dim": "via",
     "text": "At $8.99 per 10 oz bottle, how likely are you to try one?"},
    {"id": "q3", "dim": "fea",
     "text": "How confident are you that Graza can make a fresh, additive-free dressing?"},
    {"id": "q4", "dim": "des",
     "text": "Does a dressing line *fit* Graza’s fun squeeze-bottle brand?"},
    {"id": "q5", "dim": "via",
     "text": "Would Graza dressings replace or complement your current dressing?"},
    {"id": "q6", "dim": "fea",
     "text": "How feasible is the squeeze-bottle format for salad dressings?"},
]

# ------------------------------
# 3 · Pre-sampled persona responses (1-5 scale)
# ------------------------------
RESPONSES = [
    {"persona": "p01", "q1": 4.8, "q2": 4.1, "q3": 3.8, "q4": 4.9, "q5": 4.0, "q6": 4.5},
    {"persona": "p02", "q1": 3.4, "q2": 2.2, "q3": 3.1, "q4": 3.0, "q5": 2.7, "q6": 3.3},
    {"persona": "p03", "q1": 4.6, "q2": 4.0, "q3": 3.9, "q4": 4.2, "q5": 3.8, "q6": 3.7},
    {"persona": "p04", "q1": 4.3, "q2": 3.5, "q3": 3.5, "q4": 4.1, "q5": 3.6, "q6": 3.4},
    {"persona": "p05", "q1": 3.1, "q2": 3.0, "q3": 4.2, "q4": 2.9, "q5": 2.8, "q6": 3.2},
    {"persona": "p06", "q1": 4.5, "q2": 3.7, "q3": 3.6, "q4": 4.4, "q5": 3.9, "q6": 4.1},
]

# ------------------------------
# 4 · Coordinates (one per persona for instant layout)
# ------------------------------
COORDINATES = [
    {"x": 3.2, "y": 4.6},
    {"x": 2.7, "y": 3.0},
    {"x": 4.4, "y": 4.1},
    {"x": 3.8, "y": 3.7},
    {"x": 3.0, "y": 3.4},
    {"x": 4.1, "y": 4.3},
]

# ------------------------------
# Helper functions
# ------------------------------
def compute_dvf(responses: dict, prior: dict, w_prior: float):
    """Blend persona prior with questionnaire means."""
    dims = {"des": [], "via": [], "fea": []}
    for q, v in responses.items():
        dim = next(item["dim"] for item in QUESTIONS if item["id"] == q)
        dims[dim].append(v)
    mean = {d: np.mean(vs) for d, vs in dims.items()}
    return {d: w_prior * prior[d] + (1 - w_prior) * mean[d] for d in mean}

def add_variance(score: float, sigma: float):
    return float(np.clip(np.random.normal(score, sigma), 1.0, 5.0))

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("🔧 Controls")
    show_legend = st.checkbox("Show Feasibility legend", value=False)
    selected_names = st.multiselect(
        "Personas",
        [p["name"] for p in PERSONAS],
        default=[p["name"] for p in PERSONAS],
    )
    sigma = st.slider("Variance σ", 0.05, 0.50, 0.15, 0.05)
    w_prior = st.select_slider(
        "Prior weight (how much the baseline opinions matter)",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        value=0.25,
    )
    st.caption("Lower σ = more agreement across panel")

# Slice data
mask = [p["name"] in selected_names for p in PERSONAS]
personas = [p for p, m in zip(PERSONAS, mask) if m]
responses = [r for r in RESPONSES if r["persona"] in {p["id"] for p in personas}]
coords = [c for c, m in zip(COORDINATES, mask) if m]

# ------------------------------
# Tabs
# ------------------------------
tab_map, tab_det = st.tabs(["DVF Map", "Persona Details"])

# ==============================
# Tab 1 · Bubble Map
# ==============================
with tab_map:
    st.subheader("DVF Bubble Map")
    rows = []
    for p, resp, coord in zip(personas, responses, coords):
        base = compute_dvf(
            {k: v for k, v in resp.items() if k != "persona"},
            p["prior"],
            w_prior,
        )
        dvf = {d: add_variance(s, sigma) for d, s in base.items()}
        rows.append({**{"id": p["id"], "name": p["name"], **dvf}, **coord})

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="via",
        y="des",
        size="fea",
        color="fea",
        color_continuous_scale="RdYlGn_r",
        range_color=[1, 5],
        size_max=60,
        opacity=0.9,
        hover_data={"name": True, "des": ":.2f", "via": ":.2f", "fea": ":.2f"},
    )
    fig.update_traces(
        mode="markers+text",
        text=df["name"].str.split().str[0],
        textposition="middle center",
        marker=dict(line=dict(width=1, color="rgba(0,0,0,0.3)")),
    )
    if not show_legend:
        fig.update_coloraxes(showscale=False)
    fig.update_layout(
        xaxis_title="Viability (Likelihood to Purchase)",
        yaxis_title="Desirability (Consumer Appeal)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        margin=dict(l=120, r=30, t=30, b=90),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ==============================
# Tab 2 · Persona Details
# ==============================
with tab_det:
    st.subheader("Persona Details")
    sel_name = st.selectbox("Choose Persona", [p["name"] for p in personas])
    sel_p = next(p for p in personas if p["name"] == sel_name)
    sel_r = next(r for r in responses if r["persona"] == sel_p["id"])
    dvf = compute_dvf({k: v for k, v in sel_r.items() if k != "persona"}, sel_p["prior"], w_prior)

    st.markdown(f"### {sel_name}")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Desirability", f"{dvf['des']:.2f}")
        st.metric("Viability", f"{dvf['via']:.2f}")
        st.metric("Feasibility", f"{dvf['fea']:.2f}")
    with c2:
        st.write("**Baseline Prior (1-5)**")
        st.write(sel_p["prior"])
    st.divider()
    st.write("**Raw Questionnaire Answers (1-5)**")
    for q in QUESTIONS:
        st.write(f"*{q['text']}* — {sel_r[q['id']]}")

