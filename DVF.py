# segmentation_app.py
"""
Sales Segment & Messaging Assistant — Demo

Purpose
-------
Help sales teams map an individual buyer to a segment and instantly tailor
value propositions, offers, channels, and message copy.

Run locally
-----------
    pip install streamlit numpy pandas plotly
    streamlit run segmentation_app.py

Notes
-----
• This is a self-contained demo with synthetic segments, features, and buyers.
• No external services required. Feel free to swap in your own data.
"""

from __future__ import annotations
import math, re, json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Core feature model
# Scale: all feature inputs expected on 0–5 scale (float)
# ──────────────────────────────────────────────────────────────────────────────
FEATURES: List[Dict] = [
    {"id": "price_sensitivity",   "name": "Price Sensitivity",   "desc": "Focus on cost, discounts, TCO"},
    {"id": "innovation_appetite", "name": "Innovation Appetite", "desc": "Willingness to try new solutions"},
    {"id": "risk_aversion",       "name": "Risk Aversion",       "desc": "Desire for certainty, references, SLAs"},
    {"id": "decision_speed",      "name": "Decision Speed",      "desc": "Prefers fast, lightweight decisions"},
    {"id": "digital_pref",        "name": "Digital Preference",  "desc": "Self-serve, product-led behaviors"},
    {"id": "relationship_pref",   "name": "Relationship Pref.",  "desc": "Prefers consultative, high-touch"},
    {"id": "roi_horizon",         "name": "ROI Horizon",         "desc": "0=Immediate, 5=Long-term strategic"},
    {"id": "customization_need",  "name": "Customization Need",  "desc": "Integration and flexibility demand"},
    {"id": "support_expectation", "name": "Support Expectation", "desc": "Hands-on help, onboarding, SLAs"},
    {"id": "compliance_emphasis", "name": "Compliance Emphasis", "desc": "Security, privacy, audits"},
]

FEATURE_IDS = [f["id"] for f in FEATURES]

# ──────────────────────────────────────────────────────────────────────────────
# Segments (synthetic) — centroids are on 0–1 scale (we'll map sliders 0–5 → 0–1)
# ──────────────────────────────────────────────────────────────────────────────
SEGMENTS: List[Dict] = [
    {
        "id": "value_max",
        "name": "Value Maximizers",
        "centroid": {
            "price_sensitivity": .95, "innovation_appetite": .40, "risk_aversion": .55,
            "decision_speed": .60, "digital_pref": .55, "relationship_pref": .40,
            "roi_horizon": .20, "customization_need": .40, "support_expectation": .50,
            "compliance_emphasis": .40,
        },
        "value_props": [
            "Lowest TCO in category",
            "Fast time-to-value with measurable ROI",
            "Transparent, predictable pricing",
        ],
        "offers": [
            "Bundle discounts / annual save plans",
            "ROI calculator + switching guide",
            "Competitive takeout incentives",
        ],
        "channels": ["Comparison sheets", "Customer proof / ROI stories", "Email nurtures"],
    },
    {
        "id": "speed_seek",
        "name": "Speed Seekers",
        "centroid": {
            "price_sensitivity": .45, "innovation_appetite": .60, "risk_aversion": .30,
            "decision_speed": .95, "digital_pref": .95, "relationship_pref": .25,
            "roi_horizon": .35, "customization_need": .45, "support_expectation": .35,
            "compliance_emphasis": .30,
        },
        "value_props": [
            "Deploy in hours, not weeks",
            "Self-serve setup with instant trial",
            "Automation that removes manual steps",
        ],
        "offers": [
            "14‑day trial + quickstart templates",
            "One-click integrations",
            "Starter tier with usage-based pricing",
        ],
        "channels": ["Product-led flows", "In-app prompts", "Short demo videos"],
    },
    {
        "id": "risk_steward",
        "name": "Risk‑Averse Stewards",
        "centroid": {
            "price_sensitivity": .50, "innovation_appetite": .25, "risk_aversion": .95,
            "decision_speed": .30, "digital_pref": .45, "relationship_pref": .55,
            "roi_horizon": .60, "customization_need": .50, "support_expectation": .85,
            "compliance_emphasis": .95,
        },
        "value_props": [
            "Enterprise‑grade security and compliance",
            "Proven with peers + audited SLAs",
            "Migration without downtime",
        ],
        "offers": [
            "Security brief + reference calls",
            "White‑glove onboarding",
            "Risk‑free pilot with milestones",
        ],
        "channels": ["Webinars", "Solution briefs", "Executive briefings"],
    },
    {
        "id": "innovators",
        "name": "Innovators & Tinkerers",
        "centroid": {
            "price_sensitivity": .30, "innovation_appetite": .98, "risk_aversion": .20,
            "decision_speed": .70, "digital_pref": .80, "relationship_pref": .35,
            "roi_horizon": .55, "customization_need": .95, "support_expectation": .45,
            "compliance_emphasis": .35,
        },
        "value_props": [
            "Open APIs and extensibility",
            "Roadmap influence + beta access",
            "Composable building blocks",
        ],
        "offers": [
            "Developer sandbox",
            "Solution accelerators / sample code",
            "Community + office hours",
        ],
        "channels": ["Docs", "Developer community", "Hack days"],
    },
    {
        "id": "partners",
        "name": "Partnership Builders",
        "centroid": {
            "price_sensitivity": .45, "innovation_appetite": .55, "risk_aversion": .55,
            "decision_speed": .45, "digital_pref": .45, "relationship_pref": .98,
            "roi_horizon": .65, "customization_need": .70, "support_expectation": .90,
            "compliance_emphasis": .60,
        },
        "value_props": [
            "Dedicated account team + co‑planning",
            "Tailored integration to workflows",
            "Shared success metrics and QBRs",
        ],
        "offers": [
            "Workshop + value blueprint",
            "Integration sprint",
            "Executive sponsorship",
        ],
        "channels": ["In‑person workshops", "ABM plays", "Executive dinners"],
    },
    {
        "id": "enterprise_strat",
        "name": "Enterprise Strategists",
        "centroid": {
            "price_sensitivity": .50, "innovation_appetite": .45, "risk_aversion": .70,
            "decision_speed": .30, "digital_pref": .55, "relationship_pref": .60,
            "roi_horizon": .95, "customization_need": .85, "support_expectation": .85,
            "compliance_emphasis": .90,
        },
        "value_props": [
            "Multi‑year roadmap alignment",
            "Scale, governance, and control",
            "Total economic impact across org",
        ],
        "offers": [
            "Multi‑year pricing & SSO/SOC2 package",
            "Migration program",
            "Center‑of‑excellence enablement",
        ],
        "channels": ["RFP support", "Industry events", "Executive briefings"],
    },
]

SEGMENT_IDS = [s["id"] for s in SEGMENTS]
SEGMENT_BY_ID = {s["id"]: s for s in SEGMENTS}

# Synthetic example buyers (for quick demos)
SAMPLE_BUYERS: List[Dict] = [
    {
        "id": "b1",
        "name": "Founder Fiona (Seed SaaS)",
        "profile": {
            "price_sensitivity": 3.5, "innovation_appetite": 4.8, "risk_aversion": 1.5,
            "decision_speed": 4.8, "digital_pref": 4.6, "relationship_pref": 2.0,
            "roi_horizon": 2.0, "customization_need": 3.5, "support_expectation": 2.5,
            "compliance_emphasis": 1.5,
        },
        "notes": "Needs speed, self-serve trial, integrate with Slack and HubSpot.",
    },
    {
        "id": "b2",
        "name": "Procurement Priya (Global Co)",
        "profile": {
            "price_sensitivity": 3.0, "innovation_appetite": 2.5, "risk_aversion": 4.8,
            "decision_speed": 2.0, "digital_pref": 3.0, "relationship_pref": 4.2,
            "roi_horizon": 4.2, "customization_need": 3.8, "support_expectation": 4.6,
            "compliance_emphasis": 4.9,
        },
        "notes": "Security review, SOC2, references; onboarding plan; multi-year pricing.",
    },
    {
        "id": "b3",
        "name": "Ops Omar (Midsize Manufacturing)",
        "profile": {
            "price_sensitivity": 3.8, "innovation_appetite": 3.2, "risk_aversion": 3.8,
            "decision_speed": 2.8, "digital_pref": 3.2, "relationship_pref": 4.4,
            "roi_horizon": 3.5, "customization_need": 4.2, "support_expectation": 4.4,
            "compliance_emphasis": 3.6,
        },
        "notes": "Prefers workshop; needs ERP integration; cares about uptime and SLAs.",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def to01(x: float) -> float:
    return max(0.0, min(1.0, x / 5.0))


def vec_from_profile(profile_0to5: Dict[str, float]) -> np.ndarray:
    return np.array([to01(profile_0to5.get(fid, 0.0)) for fid in FEATURE_IDS], dtype=float)


def centroid_vec(segment: Dict) -> np.ndarray:
    return np.array([segment["centroid"][fid] for fid in FEATURE_IDS], dtype=float)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a); bn = np.linalg.norm(b)
    if an == 0 or bn == 0: return 0.0
    return float(np.dot(a, b) / (an * bn))


def softmax(xs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    # higher temperature → more even; lower → peakier
    t = max(1e-3, float(temperature))
    x = (xs / t) - np.max(xs / t)
    e = np.exp(x)
    return e / np.sum(e)

TEXT_CUES = [
    (r"\b(cost|budget|price|expens)\w*", {"price_sensitivity": +0.8}),
    (r"\b(speed|urgent|asap|fast)\b",   {"decision_speed": +0.8}),
    (r"\btrial|poc|self-serve|self serve|try\b", {"digital_pref": +0.7}),
    (r"\bsecurity|soc2|iso|compliance|audit\b", {"compliance_emphasis": +1.0, "risk_aversion": +0.6}),
    (r"\bcustom(ize|ization)?|integrat(e|ion)\b", {"customization_need": +0.9}),
    (r"\bsupport|onboard(ing)?|sla\b", {"support_expectation": +0.9}),
    (r"\broi|payback|savings|tco\b", {"price_sensitivity": +0.5, "roi_horizon": -0.6}),
    (r"\bworkshop|qbr|partner(ship)?\b", {"relationship_pref": +0.9}),
    (r"\binnov(ation|ate)|beta|api\b", {"innovation_appetite": +1.0}),
]


def apply_text_cues(profile: Dict[str, float], notes: str, cap: float = 5.0) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (updated_profile, deltas) after boosting features based on notes."""
    deltas = {k: 0.0 for k in FEATURE_IDS}
    if not notes: return profile, deltas
    text = notes.lower()
    for pat, bonus in TEXT_CUES:
        if re.search(pat, text):
            for k, v in bonus.items():
                before = profile.get(k, 0.0)
                profile[k] = float(min(cap, before + v))
                deltas[k] += v
    return profile, deltas


def feature_contributions(profile_vec: np.ndarray, seg_vec: np.ndarray) -> pd.DataFrame:
    """Approximate contribution per feature (signed). Higher → more supportive of the match."""
    # Center features around 0.5 to see positive vs negative tilt relative to neutral
    centered_profile = profile_vec - 0.5
    centered_seg = seg_vec - 0.5
    contrib = centered_profile * centered_seg  # elementwise alignment
    df = pd.DataFrame({
        "feature": FEATURE_IDS,
        "name": [f["name"] for f in FEATURES],
        "contribution": contrib,
        "profile": profile_vec,
        "segment": seg_vec,
    }).sort_values("contribution", ascending=False)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sales Segment & Messaging Assistant", layout="wide")
st.title("🧭 Sales Segment & Messaging Assistant — Demo")

with st.sidebar:
    st.markdown("### Inputs")
    # Choose sample buyer or custom
    sample_names = [b["name"] for b in SAMPLE_BUYERS]
    buyer_choice = st.selectbox("Start from buyer persona", ["(Manual entry)"] + sample_names)

    # Product & tone
    product_name = st.text_input("Product / Offer name", value="Your Product")
    tone = st.select_slider("Message tone", options=["Practical", "Reassuring", "Bold"], value="Practical")

    # Controls
    temperature = st.slider("Segmenting temperature", 0.3, 2.0, 0.8, 0.1,
                            help="Lower = peakier (confident). Higher = more even probabilities.")
    w_prior = st.select_slider("Persona prior weight", options=[0.0, 0.25, 0.5, 0.75, 1.0], value=0.25,
                               help="Blend between persona prior (if selected) and your entered sliders.")

    st.markdown("---")
    st.caption("Set buyer feature sliders (0–5)")

    # Initialize sliders either from manual defaults or selected persona
    init_profile = {fid: 2.5 for fid in FEATURE_IDS}
    init_notes = ""
    if buyer_choice != "(Manual entry)":
        b = next(x for x in SAMPLE_BUYERS if x["name"] == buyer_choice)
        init_profile = b["profile"].copy()
        init_notes = b.get("notes", "")

    sliders = {}
    for f in FEATURES:
        sliders[f["id"]] = st.slider(f["name"], 0.0, 5.0, float(init_profile[f["id"]]), 0.5)

    notes = st.text_area("Discovery notes (optional)", value=init_notes, height=90,
                         help="Paste notes; keywords will auto-influence the profile (e.g., 'SOC2', 'trial', 'budget').")

    show_debug = st.checkbox("Show debug tables", value=False)

# Apply persona prior blend (if any)
manual_profile = {fid: sliders[fid] for fid in FEATURE_IDS}

if buyer_choice != "(Manual entry)":
    persona = next(x for x in SAMPLE_BUYERS if x["name"] == buyer_choice)
    prior = persona["profile"].copy()
else:
    prior = {fid: 2.5 for fid in FEATURE_IDS}

blended = {fid: float(w_prior) * prior[fid] + (1.0 - float(w_prior)) * manual_profile[fid] for fid in FEATURE_IDS}

# Apply text cues
blended_after_text, text_deltas = apply_text_cues(blended.copy(), notes)

# Vectorize
p_vec = vec_from_profile(blended_after_text)

# Similarities & probabilities
sims = []
for seg in SEGMENTS:
    s_vec = centroid_vec(seg)
    sims.append(cosine_sim(p_vec, s_vec))

sims = np.array(sims)
probs = softmax(sims, temperature=temperature)

ranked = pd.DataFrame({
    "segment": [s["name"] for s in SEGMENTS],
    "id": SEGMENT_IDS,
    "similarity": sims,
    "probability": probs,
}).sort_values("probability", ascending=False).reset_index(drop=True)

# Top match
top_id = ranked.loc[0, "id"]
top_seg = SEGMENT_BY_ID[top_id]
top_vec = centroid_vec(top_seg)

# Contributions
contrib_df = feature_contributions(p_vec, top_vec)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs: Segment Match · Messaging · Offers & Channels · Drivers · Export
# ──────────────────────────────────────────────────────────────────────────────
T1, T2, T3, T4, T5 = st.tabs([
    "Segment Match", "Messaging", "Offers & Channels", "Drivers", "Export",
])

# ── Segment Match ─────────────────────────────────────────────────────────────
with T1:
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.subheader("Segment probabilities")
        prob_bar = px.bar(ranked, x="segment", y="probability", text="probability",
                          labels={"segment": "Segment", "probability": "Probability"}, height=420)
        prob_bar.update_traces(texttemplate="%{y:.2f}")
        prob_bar.update_layout(yaxis_range=[0, 1.0], plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=40,r=20,t=20,b=80))
        st.plotly_chart(prob_bar, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(ranked[["segment", "similarity", "probability"]].style.format({"similarity": "{:.3f}", "probability": "{:.2f}"}), use_container_width=True)

    with c2:
        st.subheader("Profile vs top segment")
        theta = [f["name"] for f in FEATURES]
        r_profile = [p_vec[i] for i,_ in enumerate(FEATURES)]
        r_segment = [top_vec[i] for i,_ in enumerate(FEATURES)]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=r_profile, theta=theta, fill="toself", name="Buyer"))
        radar.add_trace(go.Scatterpolar(r=r_segment, theta=theta, fill="toself", name=top_seg["name"]))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                            showlegend=True, height=420, margin=dict(l=20,r=20,t=20,b=20),
                            paper_bgcolor="white")
        st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})

    st.info(f"Top segment: **{top_seg['name']}**  \
             Rationale: alignment across {', '.join(contrib_df.head(3)['name'])}; watch-outs: {', '.join(contrib_df.tail(2)['name'])}.")

# ── Messaging ─────────────────────────────────────────────────────────────────
with T2:
    st.subheader("Key messages by funnel stage")

    def tone_wrap(text: str) -> str:
        if tone == "Bold":
            return text.replace("{{adjective}}", "unbeatable").replace("{{verb}}", "crush")
        if tone == "Reassuring":
            return text.replace("{{adjective}}", "proven").replace("{{verb}}", "de-risk")
        return text.replace("{{adjective}}", "practical").replace("{{verb}}", "simplify")

    # Compose snippets using top segment properties
    vp = top_seg["value_props"]
    msgs = {
        "Awareness": f"{{product}} helps you {{verb}} work with {{adjective}} efficiency. {vp[0]}.",
        "Consideration": f"Teams like yours choose {{product}} for: • {vp[0]} • {vp[1]} • {vp[2]}",
        "Decision": f"Move forward confidently: {vp[1]}. Get started in days with our {{adjective}} onboarding.",
    }

    for stage, template in msgs.items():
        txt = tone_wrap(template).replace("{product}", product_name).replace("{{product}}", product_name)
        st.markdown(f"**{stage}**")
        st.text_area(f"{stage} message", value=txt, height=80, key=f"msg_{stage}")

    st.markdown("---")
    st.subheader("Objections & rebuttals (auto)")
    objections = []
    if blended_after_text["compliance_emphasis"] > 3.5:
        objections.append(("Security/Compliance", "We provide SOC2 Type II, SSO/SCIM, and data residency options."))
    if blended_after_text["price_sensitivity"] > 3.5:
        objections.append(("Price/Budget", "Lowest TCO vs alternatives; ROI model shows payback < 6 months."))
    if blended_after_text["risk_aversion"] > 3.5:
        objections.append(("Risk/Change", "Pilot with clear milestones and opt-out; references available."))

    if objections:
        for k,v in objections:
            st.write(f"• **{k}:** {v}")
    else:
        st.caption("No major objections inferred from profile.")

# ── Offers & Channels ─────────────────────────────────────────────────────────
with T3:
    st.subheader("Recommended plays")

    offers_df = pd.DataFrame({
        "Type": ["Value Proposition"]*len(top_seg["value_props"]) + ["Offer"]*len(top_seg["offers"]) + ["Channel"]*len(top_seg["channels"]),
        "Item": top_seg["value_props"] + top_seg["offers"] + top_seg["channels"],
    })

    # Weight by probability of top segment and feature alignment magnitude
    strength = float(ranked.loc[0, "probability"]) * (float(contrib_df["contribution"].abs().mean()) + 0.2)
    offers_df["Priority"] = np.clip(np.round(5 * strength, 1), 1.0, 5.0)

    st.dataframe(offers_df, use_container_width=True)

    st.markdown("**Playbook highlights**")
    st.write("- Lead with: **{}**".format(top_seg["value_props"][0]))
    st.write("- Prove with: **{}**".format(top_seg["value_props"][1]))
    st.write("- CTA: **{}**".format(top_seg["offers"][0]))

# ── Drivers ───────────────────────────────────────────────────────────────────
with T4:
    st.subheader("What drove this match?")

    top_pos = contrib_df.head(5)
    top_neg = contrib_df.tail(5).iloc[::-1]

    drivers = pd.concat([
        top_pos.assign(kind="Driver"),
        top_neg.assign(kind="Friction")
    ])

    bar = px.bar(drivers, x="name", y="contribution", color="kind", height=420,
                 labels={"name": "Feature", "contribution": "Contribution"})
    bar.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=40,r=20,t=20,b=80))
    st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Show feature details"):
        detail_cols = ["name", "profile", "segment", "contribution"]
        st.dataframe(contrib_df[detail_cols].rename(columns={"name": "Feature"}).style.format({
            "profile": "{:.2f}", "segment": "{:.2f}", "contribution": "{:.3f}"
        }), use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
with T5:
    st.subheader("Export & share")
    # Collect export data
    export = {
        "product": product_name,
        "tone": tone,
        "ranked_segments": ranked.to_dict(orient="records"),
        "profile_0to5": blended_after_text,
        "text_cue_deltas": text_deltas,
        "top_segment": top_seg["name"],
        "value_props": top_seg["value_props"],
        "offers": top_seg["offers"],
        "channels": top_seg["channels"],
    }
    export_json = json.dumps(export, indent=2)
    st.download_button("⬇️ Download JSON", data=export_json, file_name="buyer_segment_recommendations.json", mime="application/json")

    # CSV for ranked segments
    csv = ranked.to_csv(index=False).encode()
    st.download_button("⬇️ Download Segment Probabilities (CSV)", data=csv, file_name="segment_probabilities.csv", mime="text/csv")

    st.markdown("**Executive summary (editable)**")
    summary = (
        f"Top segment: {top_seg['name']}. Highest alignment on "
        f"{', '.join(contrib_df.head(3)['name'])}. "
        f"Lead with: {top_seg['value_props'][0]}. Offer: {top_seg['offers'][0]}. "
        f"Primary channels: {', '.join(top_seg['channels'][:2])}."
    )
    st.text_area("Summary", value=summary, height=120)

# ──────────────────────────────────────────────────────────────────────────────
# Debug (optional)
# ──────────────────────────────────────────────────────────────────────────────
if show_debug:
    st.markdown("---")
    st.subheader("Debug")
    st.write("Profile after text cues (0–5):", blended_after_text)
    st.write("Text cue deltas:", text_deltas)
    st.write("Feature vector (0–1):", p_vec)
    st.write("Similarities:", sims)
    st.write("Probabilities:", probs)
