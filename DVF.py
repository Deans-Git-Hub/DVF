# dvf_app.py
"""
Graza Dressing DVF Scoring Assistant — Demo
Run:
    pip install streamlit pandas numpy plotly
    streamlit run dvf_app.py
"""

import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
import streamlit as st

# ─────────────────── Synthetic Personas ──────────────────────
PERSONAS = [
    {"id":"p01","name":"Social-Chef Sara",        "prior":{"des":4.5,"via":3.8,"fea":3.6},"k_price":.10},
    {"id":"p02","name":"Frugal-Dad Dan",         "prior":{"des":3.2,"via":2.6,"fea":3.3},"k_price":.25},
    {"id":"p03","name":"Clean-Eating Chloe",     "prior":{"des":4.3,"via":3.9,"fea":3.7},"k_price":.08},
    {"id":"p04","name":"Eco-Mom Maya",           "prior":{"des":4.1,"via":3.4,"fea":3.2},"k_price":.12},
    {"id":"p05","name":"Chef-Next-Door Marco",   "prior":{"des":3.0,"via":3.2,"fea":4.0},"k_price":.06},
    {"id":"p06","name":"Time-Starved Tina",      "prior":{"des":4.2,"via":3.6,"fea":3.5},"k_price":.09},
]

QUESTIONS = [
    {"id":"q1","dim":"des","text":"Appeal of Graza ready-to-use dressings"},
    {"id":"q2","dim":"via","text":"Likelihood to buy at current price"},
    {"id":"q3","dim":"fea","text":"Confidence in fresh additive-free recipe"},
    {"id":"q4","dim":"des","text":"Brand fit with squeeze bottle"},
    {"id":"q5","dim":"via","text":"Replace vs complement current dressing"},
    {"id":"q6","dim":"fea","text":"Ease of squeeze bottle in routine"},
]

RESPONSES = [
    {"persona":"p01","q1":4.8,"q2":4.1,"q3":3.8,"q4":4.9,"q5":4.0,"q6":4.5},
    {"persona":"p02","q1":3.4,"q2":2.2,"q3":3.1,"q4":3.0,"q5":2.7,"q6":3.3},
    {"persona":"p03","q1":4.6,"q2":4.0,"q3":3.9,"q4":4.2,"q5":3.8,"q6":3.7},
    {"persona":"p04","q1":4.3,"q2":3.5,"q3":3.5,"q4":4.1,"q5":3.6,"q6":3.4},
    {"persona":"p05","q1":3.1,"q2":3.0,"q3":4.2,"q4":2.9,"q5":2.8,"q6":3.2},
    {"persona":"p06","q1":4.5,"q2":3.7,"q3":3.6,"q4":4.4,"q5":3.9,"q6":4.1},
]

COORDS = [
    {"x":3.2,"y":4.6},{"x":2.6,"y":3.0},{"x":4.4,"y":4.1},
    {"x":3.4,"y":3.9},{"x":3.0,"y":3.3},{"x":4.1,"y":4.3},
]

# ─────────────── Helper functions ────────────────────────────
def piecewise_penalty(price: float) -> float:
    """Return baseline penalty in Viability points (before persona scaling)."""
    if price <= 10:
        return 0.10 * (price - 8.99)                        # gentle slope
    elif price <= 13:
        return 0.10 * (10 - 8.99) + 0.40 * (price - 10)    # sticker-shock zone
    else:
        return (0.10 * (10 - 8.99) + 0.40 * 3 +
                0.05 * (price - 13))                       # ultra-premium plateau

def dvf_from_responses(resp, persona, price, w_prior):
    dims={"des":[],"via":[],"fea":[]}
    for q,v in resp.items():
        dims[next(d["dim"] for d in QUESTIONS if d["id"]==q)].append(v)
    mean={d:np.mean(vs) for d,vs in dims.items()}

    # Apply piece-wise penalty scaled by persona sensitivity
    penalty = piecewise_penalty(price) * persona["k_price"]
    mean["via"] = np.clip(mean["via"] - penalty, 1, 5)

    return {d:w_prior*persona["prior"][d] + (1-w_prior)*mean[d] for d in dims}

rng = np.random.default_rng()
def noisy(val,sigma): return float(np.clip(rng.normal(val,sigma),1,5))

# ═══════════ Streamlit UI ════════════════════════════════════
st.set_page_config(page_title="Graza DVF",layout="wide")
st.title("🥗 Graza Dressing DVF Scoring Assistant")

# Sidebar
with st.sidebar:
    show_leg = st.checkbox("Show feasibility legend", False)
    chosen   = st.multiselect("Personas", [p["name"] for p in PERSONAS],
                              default=[p["name"] for p in PERSONAS])
    price    = st.slider("Bottle price ($)", 6.99, 14.99, 8.99, 0.50)
    sigma    = st.slider("Variance σ", 0.05, 0.50, 0.15, 0.05)
    w_prior  = st.select_slider("Prior weight", [0, .25, .5, .75, 1], value=.25)

# Build DF
panel=[]
for p,coord in zip(PERSONAS, COORDS):
    if p["name"] not in chosen: continue
    raw = next(r for r in RESPONSES if r["persona"]==p["id"])
    dvf = dvf_from_responses({k:v for k,v in raw.items() if k!="persona"}, p, price, w_prior)
    dvf = {d:noisy(v,sigma) for d,v in dvf.items()}
    panel.append({**coord, **dvf, "name":p["name"]})
df=pd.DataFrame(panel)

# Tabs: Map · Details · Price · Disagreement · Summary
tab_map, tab_det, tab_price, tab_var, tab_sum = st.tabs(
    ["Bubble Map","Persona Details","Price Sensitivity","Question Disagreement","Summary"])

# ── Bubble Map ────────────────────────────────────────────────
with tab_map:
    med_x, med_y = df["via"].median(), df["des"].median()
    fig = px.scatter(df, x="via", y="des", size="fea", color="fea",
                     color_continuous_scale="RdYlGn_r", range_color=[1,5],
                     size_max=70, opacity=.9,
                     hover_data={"name":True,"des":":.2f","via":":.2f","fea":":.2f"},
                     height=520)
    fig.update_traces(mode="markers+text",
                      text=df["name"].str.split().str[0],
                      textposition="middle center",
                      marker=dict(line=dict(width=1,color="rgba(0,0,0,0.3)")))
    if not show_leg: fig.update_coloraxes(showscale=False)
    fig.add_vline(x=med_x,line_dash="dash",line_color="gray",opacity=.4)
    fig.add_hline(y=med_y,line_dash="dash",line_color="gray",opacity=.4)
    fig.update_layout(xaxis_title="Viability", yaxis_title="Desirability",
                      plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=120,r=40,t=20,b=80))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.download_button("⬇️ CSV", df.to_csv(index=False).encode(),
                       "graza_panel.csv", "text/csv")

# ── Persona Details ──────────────────────────────────────────
with tab_det:
    sel = st.selectbox("Select persona", df["name"])
    p_row = df.set_index("name").loc[sel]
    st.markdown(f"### {sel}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Desirability", f"{p_row['des']:.2f}")
    c2.metric("Viability",   f"{p_row['via']:.2f}")
    c3.metric("Feasibility", f"{p_row['fea']:.2f}")
    raw = next(r for r in RESPONSES if r["persona"]==next(x["id"] for x in PERSONAS if x["name"]==sel))
    st.write("**Raw question scores**")
    for q in QUESTIONS:
        st.write(f"*{q['id']}* — {raw[q['id']]}")

# ── Price Sensitivity ────────────────────────────────────────
with tab_price:
    st.subheader("Average Viability vs Price")
    price_grid = np.arange(6.99, 15.0, 0.5)
    avg_via=[]
    for pr in price_grid:
        vals=[]
        for p in PERSONAS:
            if p["name"] not in chosen: continue
            resp=next(r for r in RESPONSES if r["persona"]==p["id"])
            via = dvf_from_responses({k:v for k,v in resp.items() if k!="persona"},
                                     p, pr, w_prior)["via"]
            vals.append(via)
        avg_via.append(np.mean(vals))
    line = go.Figure(go.Scatter(x=price_grid, y=avg_via, mode="lines+markers"))
    line.update_layout(xaxis_title="Bottle price ($)",
                       yaxis_title="Avg Viability (1-5)",
                       height=420, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=80,r=40,t=20,b=60))
    st.plotly_chart(line,use_container_width=True,config={"displayModeBar":False})

# ── Question Disagreement ────────────────────────────────────
with tab_var:
    st.subheader("Per-question disagreement (Std Dev)")
    wide = pd.DataFrame([{**{"name":p["name"]},
        **{q["id"]:next(r for r in RESPONSES if r["persona"]==p["id"])[q["id"]]
           for q in QUESTIONS}}
        for p in PERSONAS if p["name"] in chosen]).set_index("name")
    stds = wide.std().rename("std").reset_index()
    bar = px.bar(stds, x="index", y="std", color="std",
                 color_continuous_scale="Purples", range_color=[0,1.5],
                 labels={"index":"Question","std":"Std Dev"}, height=400)
    bar.update_layout(yaxis_range=[0,stds["std"].max()*1.15],
                      plot_bgcolor="white", paper_bgcolor="white",
                      coloraxis_showscale=False,
                      margin=dict(l=60,r=40,t=20,b=60))
    st.plotly_chart(bar,use_container_width=True,config={"displayModeBar":False})

# ── Summary ───────────────────────────────────────────────────
with tab_sum:
    st.subheader("Summary")
    k1,k2,k3 = st.columns(3)
    k1.metric("Avg Desirability", f"{df['des'].mean():.2f}")
    k2.metric("Avg Viability",    f"{df['via'].mean():.2f}")
    k3.metric("Avg Feasibility",  f"{df['fea'].mean():.2f}")
    top_des = df.nlargest(1,"des").iloc[0]
    low_via = df.nsmallest(1,"via").iloc[0]
    summary = (f"At **${price:.2f}**, desirability averages **{df['des'].mean():.2f}** "
               f"and viability **{df['via'].mean():.2f}** across the selected panel. "
               f"Strongest supporter: **{top_des['name']}** (Des {top_des['des']:.2f}). "
               f"Most price-sensitive: **{low_via['name']}** (Via {low_via['via']:.2f}). "
               "Consider introductory pricing or bundle packs to mitigate sticker shock "
               "above the $10 threshold.")
    st.text_area("Executive Brief (editable)", summary, height=120)
