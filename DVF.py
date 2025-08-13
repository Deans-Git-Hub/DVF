# graza_segments_app.py
"""
Graza Buyer Segmentation & Messaging — Demo
Axes: X = Convenience, Y = Premium / Luxury
Run:
    pip install streamlit numpy pandas plotly
    streamlit run graza_segments_app.py
"""

import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st

# ─────────────────────────── Setup ────────────────────────────
st.set_page_config(page_title="Graza Segmentation", layout="wide")
st.title("🫒 Graza Buyer Segmentation & Messaging (Demo)")

# Global feature space (0–5 scale)
FEATURES = [
    ("flavor_adventure", "Loves bold flavors / variety"),
    ("health_clean", "Prioritizes clean label / additive-free"),
    ("price_sensitivity", "Price sensitive / value seeking"),
    ("cook_freq", "Cooks at home frequently"),
    ("convenience_need", "Wants convenience & speed"),
    ("sustainability", "Eco-conscious / sustainability minded"),
    ("squeeze_affinity", "Likes squeeze bottle format"),
    ("online_grocery", "Buys groceries online"),
    ("social_hosting", "Entertains / hosts often"),
    ("novelty_seeking", "Enjoys new products / trends"),
]
FEAT_IDS = [k for k, _ in FEATURES]

# Segment dictionary (centroids + go-to-market guidance)
SEGMENTS = [
    dict(
        id="seg_foodie",
        name="Flavor-First Foodies",
        centroid={
            "flavor_adventure":4.6,"health_clean":3.4,"price_sensitivity":1.7,
            "cook_freq":4.5,"convenience_need":2.4,"sustainability":3.2,
            "squeeze_affinity":4.0,"online_grocery":2.6,"social_hosting":4.1,"novelty_seeking":4.2},
        value_props=["Single-origin flavor", "Chef-inspired recipes", "Cold extraction quality"],
        offers=["Recipe drops", "Limited runs / seasonal flavors", "Bundle: Drizzle + Sizzle"],
        channels=["IG/TikTok food creators", "YouTube recipe collabs", "In-store demos"],
        messages=[
            "Dial up dishes with peppery Drizzle & high-heat Sizzle.",
            "Chef-worthy flavor, weeknight simple."
        ],
    ),
    dict(
        id="seg_clean",
        name="Clean-Label Purists",
        centroid={
            "flavor_adventure":3.6,"health_clean":4.8,"price_sensitivity":2.1,
            "cook_freq":3.8,"convenience_need":2.6,"sustainability":3.8,
            "squeeze_affinity":3.4,"online_grocery":3.0,"social_hosting":2.8,"novelty_seeking":2.9},
        value_props=["Additive-free", "Freshness & traceability", "Transparent sourcing"],
        offers=["Trial size", "Subscribe & save (fresh rotation)", "Quality guarantee"],
        channels=["Health & wellness newsletters", "Retail endcaps w/ clean labels", "Search"],
        messages=[
            "Nothing added. Just fresh, traceable olive oil.",
            "Clean ingredients you can squeeze into every meal."
        ],
    ),
    dict(
        id="seg_value",
        name="Value-Seeking Families",
        centroid={
            "flavor_adventure":2.9,"health_clean":3.6,"price_sensitivity":4.6,
            "cook_freq":3.6,"convenience_need":3.2,"sustainability":3.0,
            "squeeze_affinity":4.2,"online_grocery":3.5,"social_hosting":2.9,"novelty_seeking":2.6},
        value_props=["Everyday quality", "Mess-free squeeze", "Stretch your dollar"],
        offers=["Family multi-pack", "Loyalty credits", "Intro discount bundles"],
        channels=["Retailer apps", "Email offers", "Instacart promos"],
        messages=[
            "Weeknight wins without the mess.",
            "Everyday olive oil that goes further."
        ],
    ),
    dict(
        id="seg_hacker",
        name="Time-Saving Meal Hackers",
        centroid={
            "flavor_adventure":3.8,"health_clean":3.6,"price_sensitivity":3.0,
            "cook_freq":2.7,"convenience_need":4.7,"sustainability":3.1,
            "squeeze_affinity":4.6,"online_grocery":4.4,"social_hosting":3.1,"novelty_seeking":3.5},
        value_props=["Squeeze-and-go convenience", "No-mess prep", "Consistent results"],
        offers=["Quick-start recipes", "Refill 2-pack", "Subscribe & save"],
        channels=["Short-form video", "Retail quick meal bays", "Food kits & DTC"],
        messages=[
            "Real flavor, zero fuss—squeeze and cook.",
            "Upgrade weeknights in one squeeze."
        ],
    ),
    dict(
        id="seg_eco",
        name="Sustainable Shoppers",
        centroid={
            "flavor_adventure":3.3,"health_clean":4.1,"price_sensitivity":2.8,
            "cook_freq":3.4,"convenience_need":3.0,"sustainability":4.7,
            "squeeze_affinity":3.7,"online_grocery":3.2,"social_hosting":3.0,"novelty_seeking":3.1},
        value_props=["Lower footprint packaging", "Responsible sourcing", "Durable bottle"],
        offers=["Refill / bulk formats", "Carbon-aware shipping", "Cause tie-ins"],
        channels=["Email + blog education", "Sustainability influencers", "Retail shelf talkers"],
        messages=[
            "Sustainably sourced, thoughtfully packaged.",
            "Better olive oil, better impact."
        ],
    ),
    dict(
        id="seg_trendy",
        name="Trendy Hosts",
        centroid={
            "flavor_adventure":4.4,"health_clean":3.5,"price_sensitivity":2.2,
            "cook_freq":3.5,"convenience_need":3.0,"sustainability":3.4,
            "squeeze_affinity":4.1,"online_grocery":3.4,"social_hosting":4.8,"novelty_seeking":4.7},
        value_props=["Giftable design", "Wow factor for guests", "Seasonal collabs"],
        offers=["Limited edition drops", "Host bundle", "Creator collabs"],
        channels=["IG Reels / Pinterest", "Pop-ups & events", "PR / earned media"],
        messages=[
            "The bottle that starts conversations.",
            "Bring the wow to your next spread."
        ],
    ),
]
SEG_NAMES = [s["name"] for s in SEGMENTS]

# ───────────────────── Utility Functions ──────────────────────
def vec_from_dict(d): return np.array([float(d[k]) for k in FEAT_IDS])

def nearest_centroid_probs(x, centroids, temp=1.2):
    dists = np.linalg.norm(centroids - x, axis=1)
    logits = -dists / max(1e-6, temp)
    e = np.exp(logits - logits.max())
    return e / e.sum(), dists

def segment_palette():
    return {s["name"]: c for s, c in zip(SEGMENTS, px.colors.qualitative.Set2)}

# ── Interpretable axes: Convenience (x) & Premium/Luxury (y) ──
W_CONV = {
    "convenience_need": 0.45,
    "online_grocery":   0.30,
    "squeeze_affinity": 0.25,
}
W_PREM = {
    "price_sensitivity_inv": 0.25,  # (5 - price_sensitivity)
    "flavor_adventure":      0.25,
    "novelty_seeking":       0.20,
    "social_hosting":        0.15,
    "health_clean":          0.10,
    "sustainability":        0.05,
}

def _weighted_sum(row, weights):
    return sum(weights[k] * row[k] for k in weights)

def business_axes_from_df(df):
    """Return (x_conv, y_prem) on 0–5 scale for a DF with the 10 feature columns."""
    df2 = df.copy()
    df2["price_sensitivity_inv"] = 5.0 - df2["price_sensitivity"]
    x = df2.apply(lambda r: _weighted_sum(r, W_CONV), axis=1)
    y = df2.apply(lambda r: _weighted_sum(r, W_PREM), axis=1)
    return x.values, y.values

def business_axes_from_centroids(centroids_dicts):
    rows = []
    for c in centroids_dicts:
        rows.append({k: float(c[k]) for k in FEAT_IDS})
    df = pd.DataFrame(rows)
    return business_axes_from_df(df)

# ─────────────────────── Sidebar Inputs ───────────────────────
with st.sidebar:
    st.header("Controls")
    n_syn = st.slider("Synthetic buyers", 30, 400, 120, 10)
    seed = st.slider("Random seed", 0, 9999, 1234, 1)
    spread = st.slider("Segment spread (σ)", 0.10, 1.00, 0.35, 0.05)
    temp = st.slider("Segmentation temperature", 0.6, 3.0, 1.2, 0.1,
                     help="Higher = softer probabilities")
    show_density = st.checkbox("Show density contours", True)
    st.markdown("---")
    st.markdown("**Add a new buyer** (optional)")
    nb = {}
    for fid, label in FEATURES:
        nb[fid] = st.slider(label, 0.0, 5.0, 3.0, 0.1, key=f"new_{fid}")
    add_btn = st.button("➕ Add buyer to cohort")

# ─────────────────── Synthetic Buyer Cohort ───────────────────
rng = np.random.default_rng(seed)
C = np.vstack([vec_from_dict(s["centroid"]) for s in SEGMENTS])  # centroid matrix

buyers, labels = [], []
for _ in range(n_syn):
    sidx = rng.integers(0, len(SEGMENTS))
    base = C[sidx]
    sample = rng.normal(base, spread, size=len(FEAT_IDS))
    sample = np.clip(sample, 0, 5)
    buyers.append(sample)
    labels.append(SEGMENTS[sidx]["name"])
buyers = np.array(buyers)

# session_state cohort (for added buyers)
if "cohort" not in st.session_state:
    st.session_state.cohort = []
if add_btn:
    st.session_state.cohort.append(dict(profile=nb.copy()))

cohort_rows = []
for i, row in enumerate(st.session_state.cohort):
    x = vec_from_dict(row["profile"])
    probs, _ = nearest_centroid_probs(x, C, temp=temp)
    seg_idx = int(np.argmax(probs))
    seg_name = SEGMENTS[seg_idx]["name"]
    cohort_rows.append({"id": f"user_{i+1}", "segment": seg_name, **row["profile"]})
cohort_df = pd.DataFrame(cohort_rows) if cohort_rows else pd.DataFrame(columns=["id","segment",*FEAT_IDS])

# ────────────────────────── Tabs ──────────────────────────────
tab_map, tab_people, tab_explore, tab_library, tab_data = st.tabs(
    ["Segmentation Map", "Individuals", "Buyer Explorer", "Offers & Messaging", "Data & Export"]
)

# ───────────────────── Segmentation Map ───────────────────────
with tab_map:
    st.subheader("Where buyers fall (X: Convenience, Y: Premium / Luxury)")

    # Base DF for plotting
    df_syn = pd.DataFrame(buyers, columns=FEAT_IDS)
    df_syn["segment"] = labels
    df_syn["type"] = "Synthetic"
    if not cohort_df.empty:
        df_add = cohort_df.copy()
        df_add["type"] = "Added"
        df_plot = pd.concat([df_syn, df_add[["segment", *FEAT_IDS, "type"]]], ignore_index=True)
    else:
        df_plot = df_syn

    # Segment centroids DataFrame
    df_cent = pd.DataFrame([s["centroid"] for s in SEGMENTS])
    df_cent["segment"] = [s["name"] for s in SEGMENTS]

    # Compute business axes for buyers & centroids
    x_buy, y_buy = business_axes_from_df(df_plot[FEAT_IDS])
    x_cent, y_cent = business_axes_from_centroids([s["centroid"] for s in SEGMENTS])

    df_plot["x"] = x_buy
    df_plot["y"] = y_buy
    df_cent["x"] = x_cent
    df_cent["y"] = y_cent

    # Convenience / Premium scores for hover
    df_plot["Convenience"] = df_plot["x"]
    df_plot["Premium/Luxury"] = df_plot["y"]

    fig = px.scatter(
        df_plot, x="x", y="y", color="segment", symbol="type",
        opacity=0.9,
        labels={"x": "Convenience", "y": "Premium / Luxury"},
        color_discrete_map=segment_palette(), height=620,
        hover_data={"segment": True, "x":":.2f", "y":":.2f", "type": True,
                    "Convenience":":.2f","Premium/Luxury":":.2f"},
    )

    # Segment centroids as anchors
    fig.add_trace(go.Scatter(
        x=df_cent["x"], y=df_cent["y"], mode="markers+text",
        text=[n.split()[0] for n in df_cent["segment"]],
        textposition="top center",
        marker=dict(size=18, line=dict(width=2, color="white"), symbol="diamond"),
        name="Segment centroid", hoverinfo="skip", showlegend=True
    ))

    # Density overlay (optional)
    if show_density:
        fig.add_trace(go.Histogram2dContour(
            x=df_plot["x"], y=df_plot["y"],
            ncontours=6, showscale=False, colorscale="Greys", opacity=0.20,
            name="Density"
        ))

    # Crosshairs & quadrant labels
    mx, my = float(df_plot["x"].median()), float(df_plot["y"].median())
    fig.add_vline(mx, line_color="rgba(0,0,0,.15)")
    fig.add_hline(my, line_color="rgba(0,0,0,.15)")
    padx = (df_plot["x"].max() - df_plot["x"].min()) * 0.04
    pady = (df_plot["y"].max() - df_plot["y"].min()) * 0.04
    for x_, y_, txt in [
        (mx+padx, my+pady, "Effortless Luxury"),
        (mx-padx, my+pady, "Culinary Experience"),
        (mx+padx, my-pady, "Quick & Practical"),
        (mx-padx, my-pady, "Traditional Value"),
    ]:
        fig.add_annotation(x=x_, y=y_, text=txt, showarrow=False, opacity=0.7)

    fig.update_layout(
        xaxis=dict(range=[0,5]),
        yaxis=dict(range=[0,5]),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=30, t=20, b=60),
        legend=dict(orientation="h", y=1.05, x=0)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption("Convenience = speed, online, squeeze. Premium/Luxury = flavor/novelty/hosting, clean/sustainable, lower price sensitivity.")

# ─────────────────────── Individuals ──────────────────────────
with tab_people:
    st.subheader("Describe the individuals → see where they land")
    # Summarize a subset of synthetic buyers
    rows = []
    for i in range(min(10, len(labels))):
        x = buyers[i]
        probs, _ = nearest_centroid_probs(x, C, temp=temp)
        seg_idx = int(np.argmax(probs))
        seg_name = SEGMENTS[seg_idx]["name"]
        conv, prem = business_axes_from_df(pd.DataFrame([dict(zip(FEAT_IDS, x))]))
        rows.append(dict(
            Buyer=f"Buyer {i+1}",
            Segment=seg_name,
            Convenience=f"{conv[0]:.2f}",
            Premium_Luxury=f"{prem[0]:.2f}",
        ))
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("This table summarizes a subset of buyers. Explore any buyer in detail in the next tab.")

# ─────────────────────── Buyer Explorer ───────────────────────
with tab_explore:
    st.subheader("Classify a single buyer & tailor the playbook")
    mode = st.radio("Select source", ["Synthetic", "Added (from sidebar)"], horizontal=True)
    if mode == "Synthetic":
        idx = st.slider("Pick synthetic buyer", 1, len(labels), 1)
        x = buyers[idx-1]
    else:
        if cohort_df.empty:
            st.info("No added buyers yet. Use the sidebar to add one.")
            x = None
        else:
            options = list(range(1, len(cohort_df)+1))
            sel = st.selectbox("Pick added buyer", options, index=0)
            x = cohort_df[FEAT_IDS].iloc[sel-1].values

    if x is not None:
        probs, _ = nearest_centroid_probs(x, C, temp=temp)
        order = np.argsort(probs)[::-1]
        top = order[0]
        winner = SEGMENTS[top]

        prob_df = pd.DataFrame({
            "Segment":[SEGMENTS[i]["name"] for i in order],
            "Probability":[float(probs[i]) for i in order]
        })
        bar = px.bar(prob_df, x="Probability", y="Segment", orientation="h",
                     range_x=[0,1], height=320, text=[f"{p*100:.0f}%" for p in prob_df["Probability"]],
                     color="Probability", color_continuous_scale="Blues")
        bar.update_layout(coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=80,r=30,t=20,b=40))
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar":False})

        # Convenience & Premium metrics
        conv, prem = business_axes_from_df(pd.DataFrame([dict(zip(FEAT_IDS, x))]))
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("#### 🧭 Recommended segment")
            st.metric(winner["name"], f"{probs[top]*100:.0f}% match")
        with c2:
            st.markdown("#### ⚡ Convenience")
            st.metric("Score (0–5)", f"{conv[0]:.2f}")
        with c3:
            st.markdown("#### 💎 Premium / Luxury")
            st.metric("Score (0–5)", f"{prem[0]:.2f}")

        st.markdown("#### 💎 Value props to emphasize")
        for v in winner["value_props"]:
            st.write(f"- {v}")

        st.markdown("#### 📣 Channels to prioritize")
        for ch in winner["channels"]:
            st.write(f"- {ch}")

        st.markdown("#### ✍️ Messaging")
        for m in winner["messages"]:
            st.write(f"• {m}")

        st.markdown("#### 💡 Offer ideas")
        st.write(", ".join(winner["offers"]))

# ─────────────────── Offers & Messaging Library ───────────────
with tab_library:
    st.subheader("Segment playbooks")
    for seg in SEGMENTS:
        with st.expander(f"{seg['name']}"):
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown("**Value props**")
                for v in seg["value_props"]:
                    st.write(f"- {v}")
            with c2:
                st.markdown("**Offers**")
                for o in seg["offers"]:
                    st.write(f"- {o}")
            with c3:
                st.markdown("**Channels**")
                for ch in seg["channels"]:
                    st.write(f"- {ch}")
            st.markdown("**Messaging examples**")
            for m in seg["messages"]:
                st.write(f"• {m}")

# ─────────────────────── Data & Export ────────────────────────
with tab_data:
    st.subheader("Current cohort data")
    syn_df = pd.DataFrame(buyers, columns=FEAT_IDS)
    # classify all for table
    segs, convs, prems = [], [], []
    for i in range(len(syn_df)):
        p,_ = nearest_centroid_probs(buyers[i], C, temp=temp)
        segs.append(SEGMENTS[int(np.argmax(p))]["name"])
        c,prem = business_axes_from_df(pd.DataFrame([dict(zip(FEAT_IDS, buyers[i]))]))
        convs.append(c[0]); prems.append(prem[0])
    syn_df.insert(0, "segment", segs)
    syn_df.insert(1, "convenience", [f"{v:.2f}" for v in convs])
    syn_df.insert(2, "premium_luxury", [f"{v:.2f}" for v in prems])
    syn_df.insert(0, "id", [f"buyer_{i+1}" for i in range(len(syn_df))])

    if not cohort_df.empty:
        # add computed axes for added buyers
        c_add, p_add = business_axes_from_df(cohort_df[FEAT_IDS])
        export_df = cohort_df.copy()
        export_df.insert(1, "convenience", [f"{v:.2f}" for v in c_add])
        export_df.insert(2, "premium_luxury", [f"{v:.2f}" for v in p_add])
        export_df = pd.concat([syn_df, export_df.reset_index(drop=True)], ignore_index=True, sort=False)
    else:
        export_df = syn_df

    st.dataframe(export_df, use_container_width=True, height=380)
    st.download_button("⬇️ Download CSV", export_df.to_csv(index=False).encode(),
                       "graza_segmented_buyers.csv", "text/csv")
    st.caption("Columns are 10 features (0–5), inferred segment labels, and the composite axes.")
