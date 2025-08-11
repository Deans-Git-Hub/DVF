# dvf_app.py
"""
Graza Dressing DVF Scoring Assistant — Demo
Run:
    pip install streamlit pandas numpy plotly
    streamlit run dvf_app.py
"""

import io, numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────── 1 · Personas ────────────────────────────
PERSONAS = [
    {"id": "p01", "name": "Social-Chef Sara",
     "prior": {"des": 4.5, "via": 3.8, "fea": 3.6}, "k_price": 0.10},
    {"id": "p02", "name": "Frugal-Dad Dan",
     "prior": {"des": 3.2, "via": 2.6, "fea": 3.3}, "k_price": 0.25},
    {"id": "p03", "name": "Clean-Eating Chloe",
     "prior": {"des": 4.3, "via": 3.9, "fea": 3.7}, "k_price": 0.08},
    {"id": "p04", "name": "Eco-Mom Maya",
     "prior": {"des": 4.1, "via": 3.4, "fea": 3.2}, "k_price": 0.12},
    {"id": "p05", "name": "Chef-Next-Door Marco",
     "prior": {"des": 3.0, "via": 3.2, "fea": 4.0}, "k_price": 0.06},
    {"id": "p06", "name": "Time-Starved Tina",
     "prior": {"des": 4.2, "via": 3.6, "fea": 3.5}, "k_price": 0.09},
]

# ─────────────────── 2 · Questionnaire ──────────────────────
QUESTIONS = [
    {"id": "q1", "dim": "des", "text": "Appeal of Graza ready-to-use dressings"},
    {"id": "q2", "dim": "via", "text": "Likelihood to buy at current price"},
    {"id": "q3", "dim": "fea", "text": "Confidence in fresh additive-free recipe"},
    {"id": "q4", "dim": "des", "text": "Brand fit with squeeze bottle"},
    {"id": "q5", "dim": "via", "text": "Replace vs complement current dressing"},
    {"id": "q6", "dim": "fea", "text": "Ease of squeeze bottle in routine"},
]

# ─────────────────── 3 · Demo responses (1-5) ────────────────
RESPONSES = [
    {"persona":"p01","q1":4.8,"q2":4.1,"q3":3.8,"q4":4.9,"q5":4.0,"q6":4.5},
    {"persona":"p02","q1":3.4,"q2":2.2,"q3":3.1,"q4":3.0,"q5":2.7,"q6":3.3},
    {"persona":"p03","q1":4.6,"q2":4.0,"q3":3.9,"q4":4.2,"q5":3.8,"q6":3.7},
    {"persona":"p04","q1":4.3,"q2":3.5,"q3":3.5,"q4":4.1,"q5":3.6,"q6":3.4},
    {"persona":"p05","q1":3.1,"q2":3.0,"q3":4.2,"q4":2.9,"q5":2.8,"q6":3.2},
    {"persona":"p06","q1":4.5,"q2":3.7,"q3":3.6,"q4":4.4,"q5":3.9,"q6":4.1},
]

# ─────────────────── 4 · Pre-laid map coords ────────────────
COORDS = [
    {"x":3.2,"y":4.6},{"x":2.6,"y":3.0},{"x":4.4,"y":4.1},
    {"x":3.4,"y":3.9},{"x":3.0,"y":3.3},{"x":4.1,"y":4.3},
]

# ─────────────────── Helpers ─────────────────────────────────
def dvf_from_responses(resp, persona, price, w_prior):
    """Return dict des/ via/ fea after price & prior blending."""
    dims={"des":[],"via":[],"fea":[]}
    for q,v in resp.items():
        dim=next(d["dim"] for d in QUESTIONS if d["id"]==q)
        dims[dim].append(v)
    mean={d:np.mean(vs) for d,vs in dims.items()}
    penalty=(price-8.99)*persona["k_price"]
    mean["via"]=np.clip(mean["via"]-penalty,1,5)
    return {d:w_prior*persona["prior"][d]+(1-w_prior)*mean[d] for d in dims}

def add_sigma(val,sig): return float(np.clip(np.random.normal(val,sig),1,5))

# ═══════════ UI ══════════════════════════════════════════════
st.set_page_config(page_title="Graza DVF",layout="wide")
st.title("🥗 Graza Dressing DVF Scoring Assistant")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    show_leg = st.checkbox("Show feasibility legend",False)
    chosen   = st.multiselect("Personas",
                [p["name"] for p in PERSONAS],
                default=[p["name"] for p in PERSONAS])
    price    = st.slider("Bottle price ($)",6.99,12.99,8.99,0.5)
    sigma    = st.slider("Variance σ",0.05,0.5,0.15,0.05)
    w_prior  = st.select_slider("Prior weight",[0,.25,.5,.75,1],value=.25)
    st.caption("σ adds random disagreement each run")

# ── Build panel DF ───────────────────────────────────────────
rows=[]
for p,coord in zip(PERSONAS,COORDS):
    if p["name"] not in chosen: continue
    resp=next(r for r in RESPONSES if r["persona"]==p["id"])
    dvf=dvf_from_responses({k:v for k,v in resp.items() if k!="persona"},p,price,w_prior)
    dvf_noisy={d:add_sigma(v,sigma) for d,v in dvf.items()}
    rows.append({**{"name":p["name"],**dvf_noisy},**coord})
df=pd.DataFrame(rows)

# ── KPI cards & brief ────────────────────────────────────────
k1,k2,k3=st.columns(3)
k1.metric("Avg Des",f"{df['des'].mean():.2f}")
k2.metric("Avg Via",f"{df['via'].mean():.2f}")
k3.metric("Avg Fea",f"{df['fea'].mean():.2f}")

best=df.nlargest(1,"des").iloc[0]
risk=df.nsmallest(1,"via").iloc[0]
brief=(f"At **${price:.2f}**, avg desirability is **{df['des'].mean():.2f}** and "
       f"viability **{df['via'].mean():.2f}**. _{best['name']}_ is your top advocate; "
       f"price pressure is greatest for _{risk['name']}_ (Via {risk['via']:.2f}).")
st.text_area("Executive Brief",brief,height=80)

# ── Tabs ─────────────────────────────────────────────────────
tab_map, tab_price, tab_var = st.tabs(
    ["Bubble Map","Price Sensitivity","Question Disagreement"])

# ── Bubble map tab ───────────────────────────────────────────
with tab_map:
    med_x, med_y = df["via"].median(), df["des"].median()
    fig=px.scatter(df,x="via",y="des",size="fea",color="fea",
                   color_continuous_scale="RdYlGn_r",range_color=[1,5],
                   size_max=70,opacity=.9,
                   hover_data={"name":True,"des":":.2f","via":":.2f","fea":":.2f"},
                   height=520)
    fig.update_traces(mode="markers+text",
        text=df["name"].str.split().str[0],textposition="middle center",
        marker=dict(line=dict(width=1,color="rgba(0,0,0,0.3)")))
    if not show_leg: fig.update_coloraxes(showscale=False)
    fig.add_vline(x=med_x,line_dash="dash",line_color="gray",opacity=.4)
    fig.add_hline(y=med_y,line_dash="dash",line_color="gray",opacity=.4)
    for txt,xmul,ymul in [("Leverage",0.8,1.05),("Fix first",0.8,0.85),
                          ("Activate",1.2,1.05),("Low impact",1.2,0.85)]:
        fig.add_annotation(x=med_x*xmul,y=med_y*ymul,text=txt,
                           showarrow=False,opacity=.6)
    fig.update_layout(xaxis_title="Viability",yaxis_title="Desirability",
                      plot_bgcolor="white",paper_bgcolor="white",
                      margin=dict(l=120,r=40,t=20,b=80))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    csv=df.to_csv(index=False).encode()
    st.download_button("⬇️ CSV",csv,"graza_panel.csv","text/csv")

# ── Price sensitivity tab ────────────────────────────────────
with tab_price:
    st.subheader("Average Viability vs Price")
    price_range=np.arange(6.99,13.0,0.5)
    series=[]
    for pr in price_range:
        vals=[]
        for p in PERSONAS:
            if p["name"] not in chosen: continue
            resp=next(r for r in RESPONSES if r["persona"]==p["id"])
            via=dvf_from_responses({k:v for k,v in resp.items() if k!="persona"},
                                   p,pr,w_prior)["via"]
            vals.append(via)
        series.append(np.mean(vals))
    line=go.Figure(go.Scatter(x=price_range,y=series,mode="lines+markers"))
    line.update_layout(xaxis_title="Bottle price ($)",
                       yaxis_title="Avg Viability (1-5)",
                       height=430,plot_bgcolor="white",paper_bgcolor="white",
                       margin=dict(l=80,r=40,t=20,b=60))
    st.plotly_chart(line,use_container_width=True,config={"displayModeBar":False})

# ── Disagreement tab (std-dev bars) ───────────────────────────
with tab_var:
    st.subheader("Which questions split the panel?")
    # build wide DF of chosen personas
    rows=[]
    for p in PERSONAS:
        if p["name"] not in chosen: continue
        resp=next(r for r in RESPONSES if r["persona"]==p["id"])
        rows.append({"name":p["name"],**{qid:resp[qid] for qid in [q["id"] for q in QUESTIONS]}})
    wide=pd.DataFrame(rows).set_index("name")
    stds=wide.std().rename("std").reset_index().rename(columns={"index":"qid"})
    bar=px.bar(stds,x="qid",y="std",color="std",color_continuous_scale="Purples",
               range_color=[0,1.5],labels={"std":"Std Dev"},
               height=400)
    bar.update_layout(yaxis_range=[0,stds["std"].max()*1.2],
                      plot_bgcolor="white",paper_bgcolor="white",
                      coloraxis_showscale=False,
                      margin=dict(l=60,r=40,t=20,b=60))
    st.plotly_chart(bar,use_container_width=True,config={"displayModeBar":False})

