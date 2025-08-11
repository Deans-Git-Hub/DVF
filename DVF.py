# dvf_app.py
"""
Graza Dressing DVF Scoring Assistant — Demo Mode
Run:
    pip install streamlit pandas numpy plotly kaleido
    streamlit run dvf_app.py
"""

import io, datetime, numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────
# 1 · Synthetic Personas  (includes price-sensitivity coeff k$)
# ──────────────────────────────────────────────────────────────
PERSONAS = [
    {"id": "p01", "name": "Social-Chef Sara",
     "prior": {"des": 4.5, "via": 3.8, "fea": 3.6}, "k_price": .10},
    {"id": "p02", "name": "Frugal-Dad Dan",
     "prior": {"des": 3.2, "via": 2.6, "fea": 3.3}, "k_price": .25},
    {"id": "p03", "name": "Clean-Eating Chloe",
     "prior": {"des": 4.3, "via": 3.9, "fea": 3.7}, "k_price": .08},
    {"id": "p04", "name": "Eco-Mom Maya",
     "prior": {"des": 4.1, "via": 3.4, "fea": 3.2}, "k_price": .12},
    {"id": "p05", "name": "Chef-Next-Door Marco",
     "prior": {"des": 3.0, "via": 3.2, "fea": 4.0}, "k_price": .06},
    {"id": "p06", "name": "Time-Starved Tina",
     "prior": {"des": 4.2, "via": 3.6, "fea": 3.5}, "k_price": .09},
]

# ──────────────────────────────────────────────────────────────
# 2 · Dressings-specific DVF questionnaire (6 items)
# ──────────────────────────────────────────────────────────────
QUESTIONS = [
    {"id": "q1", "dim": "des",
     "text": "Appeal of ready-to-use Graza olive-oil dressings"},
    {"id": "q2", "dim": "via",
     "text": "Likelihood to buy at current price"},
    {"id": "q3", "dim": "fea",
     "text": "Confidence in fresh additive-free recipe"},
    {"id": "q4", "dim": "des",
     "text": "Brand fit with squeeze-bottle format"},
    {"id": "q5", "dim": "via",
     "text": "Replace vs complement current dressing"},
    {"id": "q6", "dim": "fea",
     "text": "Ease of squeeze bottle in routine"},
]

# ──────────────────────────────────────────────────────────────
# 3 · Baseline responses (1–5) — synthetic demo data
# ──────────────────────────────────────────────────────────────
RESPONSES = [
    {"persona": "p01", "q1": 4.8, "q2": 4.1, "q3": 3.8, "q4": 4.9, "q5": 4.0, "q6": 4.5},
    {"persona": "p02", "q1": 3.4, "q2": 2.2, "q3": 3.1, "q4": 3.0, "q5": 2.7, "q6": 3.3},
    {"persona": "p03", "q1": 4.6, "q2": 4.0, "q3": 3.9, "q4": 4.2, "q5": 3.8, "q6": 3.7},
    {"persona": "p04", "q1": 4.3, "q2": 3.5, "q3": 3.5, "q4": 4.1, "q5": 3.6, "q6": 3.4},
    {"persona": "p05", "q1": 3.1, "q2": 3.0, "q3": 4.2, "q4": 2.9, "q5": 2.8, "q6": 3.2},
    {"persona": "p06", "q1": 4.5, "q2": 3.7, "q3": 3.6, "q4": 4.4, "q5": 3.9, "q6": 4.1},
]

# ──────────────────────────────────────────────────────────────
# 4 · Pre-laid coordinates (same order as PERSONAS)
# ──────────────────────────────────────────────────────────────
COORDS = [
    {"x": 3.2, "y": 4.6},
    {"x": 2.6, "y": 3.0},
    {"x": 4.4, "y": 4.1},
    {"x": 3.4, "y": 3.9},
    {"x": 3.0, "y": 3.3},
    {"x": 4.1, "y": 4.3},
]

# ══════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════
st.set_page_config(layout="wide")

# ── Sidebar controls ──────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Controls")
    show_legend = st.checkbox("Show Feasibility legend", False)
    chosen = st.multiselect("Personas", [p["name"] for p in PERSONAS],
                            default=[p["name"] for p in PERSONAS])
    price = st.slider("Bottle Price ($)", 6.99, 12.99, 8.99, 0.50)
    sigma = st.slider("Variance σ", 0.05, 0.50, 0.15, 0.05)
    w_prior = st.select_slider("Prior weight", [0, .25, .5, .75, 1], value=.25)
    st.caption("Higher σ ⇒ more disagreement")

# ── Helpers ───────────────────────────────────────────────────
def compute_dvf(resp, persona, price):
    """Blend prior with response means; adjust Viability for price."""
    price_penalty = (price - 8.99) * persona["k_price"]  # $ diff × coefficient
    dims = {"des": [], "via": [], "fea": []}
    for q, v in resp.items():
        dim = next(d["dim"] for d in QUESTIONS if d["id"] == q)
        dims[dim].append(v)
    mean = {d: np.mean(vs) for d, vs in dims.items()}
    mean["via"] = np.clip(mean["via"] - price_penalty, 1, 5)  # viability adjusted
    return {d: w_prior * persona["prior"][d] + (1 - w_prior) * mean[d] for d in dims}

def add_noise(val):  # variance simulation
    return float(np.clip(np.random.normal(val, sigma), 1, 5))

# Slice selections
sel_map = {p["name"]: p for p in PERSONAS if p["name"] in chosen}
rows = []
for p, base_resp, coord in zip(PERSONAS, RESPONSES, COORDS):
    if p["name"] not in chosen:
        continue
    dvf = compute_dvf({k: v for k, v in base_resp.items() if k != "persona"}, p, price)
    noisy = {dim: add_noise(v) for dim, v in dvf.items()}
    rows.append({**{"name": p["name"], **noisy}, **coord})

df = pd.DataFrame(rows)

# ── KPI cards & Executive brief ───────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("Avg Desirability", f"{df['des'].mean():.2f}")
k2.metric("Avg Viability",    f"{df['via'].mean():.2f}")
k3.metric("Avg Feasibility",  f"{df['fea'].mean():.2f}")

best = df.sort_values("des", ascending=False).iloc[0]
risk = df.sort_values("via").iloc[0]
brief = (
    f"At **${price:.2f}**, overall desirability is **{df['des'].mean():.2f}** while "
    f"viability averages **{df['via'].mean():.2f}**. "
    f"Top advocate: **{best['name']}** (Des {best['des']:.2f}). "
    f"Price friction: **{risk['name']}** (Via {risk['via']:.2f}). "
    f"Recommendation: target launch offers/promos to move viability above 4.0."
)
st.text_area("Executive Brief", brief, height=90)

# ── Main tabs ─────────────────────────────────────────────────
tab_map, tab_var = st.tabs(["Bubble Map", "Variance Heat-map"])

# ── Bubble Map tab ────────────────────────────────────────────
with tab_map:
    med_via, med_des = df["via"].median(), df["des"].median()

    fig = px.scatter(
        df, x="via", y="des",
        size="fea", color="fea",
        color_continuous_scale="RdYlGn_r", range_color=[1, 5],
        size_max=70, opacity=0.9,
        hover_data={"name": True, "des":":.2f","via":":.2f","fea":":.2f"},
        height=550
    )
    fig.update_traces(mode="markers+text",
                      text=df["name"].str.split().str[0],
                      textposition="middle center",
                      marker=dict(line=dict(width=1, color="rgba(0,0,0,0.3)")))
    if not show_legend:
        fig.update_coloraxes(showscale=False)

    # quadrant lines + labels
    fig.add_vline(x=med_via, line_dash="dash", line_color="gray", opacity=.4)
    fig.add_hline(y=med_des, line_dash="dash", line_color="gray", opacity=.4)
    fig.add_annotation(x=med_via*0.8, y=med_des*1.05, text="Leverage", showarrow=False, opacity=.6)
    fig.add_annotation(x=med_via*0.8, y=med_des*0.85, text="Fix First", showarrow=False, opacity=.6)
    fig.add_annotation(x=med_via*1.2, y=med_des*1.05, text="Activate", showarrow=False, opacity=.6)
    fig.add_annotation(x=med_via*1.2, y=med_des*0.85, text="Low Impact", showarrow=False, opacity=.6)

    fig.update_layout(
        xaxis_title="Viability (Purchase Likelihood)",
        yaxis_title="Desirability (Appeal)",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=120, r=40, t=20, b=80)
    )
    # render
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # download buttons (CSV + PNG)
    csv_buf = df.to_csv(index=False).encode()
    st.download_button("⬇️ CSV", csv_buf, file_name="graza_dvf_panel.csv",
                       mime="text/csv", key="csv_dl")
    # png via Kaleido
    try:
        img_bytes = fig.to_image(format="png", scale=2, engine="kaleido")
        st.download_button("⬇️ PNG", img_bytes, "bubble_map.png",
                           "image/png", key="png_dl")
    except Exception:
        st.info("Install **kaleido** for PNG download: `pip install kaleido`")

# ── Variance Heat-map tab ─────────────────────────────────────
with tab_var:
    st.subheader("Question-level disagreement")
    # build matrix personas × questions (raw 1-5)
    frame = []
    for p in PERSONAS:
        if p["name"] not in chosen:
            continue
        row = {"name": p["name"]}
        resp = next(r for r in RESPONSES if r["persona"] == p["id"])
        row.update({q["id"]: resp[q["id"]] for q in QUESTIONS})
        frame.append(row)
    mat = pd.DataFrame(frame).set_index("name")
    heat = go.Figure(data=go.Heatmap(
        z=mat.values, x=[q["id"] for q in QUESTIONS], y=mat.index,
        colorscale="Purples", zmin=1, zmax=5,
        hovertemplate="Persona: %{y}<br>Q: %{x}<br>Score: %{z}<extra></extra>",
        colorbar=dict(title="Score (1-5)")
    ))
    heat.update_layout(height=420, margin=dict(l=120, r=40, t=20, b=60),
                       xaxis_title="Question ID", yaxis_title="")
    st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})
