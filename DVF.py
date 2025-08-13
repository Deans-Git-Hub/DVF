# graza_segments_app.py
"""
Graza Buyer Segmentation & Messaging — Demo
Run:
    pip install streamlit numpy pandas plotly scikit-learn
    streamlit run graza_segments_app.py
"""

import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from sklearn.decomposition import PCA
from dataclasses import dataclass
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

SEG_LOOKUP = {s["id"]: s for s in SEGMENTS}
SEG_NAMES = [s["name"] for s in SEGMENTS]
FEAT_IDS = [k for k, _ in FEATURES]

# ───────────────────── Utility Functions ──────────────────────
def vec_from_dict(d):
    return np.array([float(d[k]) for k in FEAT_IDS])

def nearest_centroid_probs(x, centroids, temp=1.2):
    # distance → probability via softmax on negative distance
    dists = np.linalg.norm(centroids - x, axis=1)
    logits = -dists / max(1e-6, temp)
    expv = np.exp(logits - logits.max())
    return expv / expv.sum(), dists

def make_pca(centroids, buyers):
    X = np.vstack([centroids, buyers])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X)
    return pca

def project_2d(pca, X):
    Z = pca.transform(X)
    # normalize for pretty plotting
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return Z

def segment_palette():
    return {s["name"]: c for s, c in zip(SEGMENTS, px.colors.qualitative.Set2)}

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

# build centroid matrix
C = np.vstack([vec_from_dict(s["centroid"]) for s in SEGMENTS])

# sample synthetic buyers around centroids
buyers = []
labels = []
for i in range(n_syn):
    # pick a segment, then sample a buyer around its centroid
    s = rng.integers(0, len(SEGMENTS))
    base = C[s]
    sample = rng.normal(base, spread, size=len(FEAT_IDS))
    sample = np.clip(sample, 0, 5)
    buyers.append(sample)
    labels.append(SEGMENTS[s]["name"])

buyers = np.array(buyers)

# session_state cohort (for added buyers)
if "cohort" not in st.session_state:
    st.session_state.cohort = []

if add_btn:
    st.session_state.cohort.append(dict(profile=nb.copy()))

# materialize cohort into arrays
cohort_rows = []
for i, row in enumerate(st.session_state.cohort):
    x = vec_from_dict(row["profile"])
    probs, dists = nearest_centroid_probs(x, C, temp=temp)
    seg_idx = int(np.argmax(probs))
    seg_name = SEGMENTS[seg_idx]["name"]
    cohort_rows.append({"id": f"user_{i+1}", "segment": seg_name, **row["profile"]})

cohort_df = pd.DataFrame(cohort_rows) if cohort_rows else pd.DataFrame(columns=["id","segment",*FEAT_IDS])

# combined dataset: synthetic + (optional) cohort
combined = buyers
combined_labels = labels
if not cohort_df.empty:
    extra = cohort_df[FEAT_IDS].values
    combined = np.vstack([combined, extra])
    combined_labels = combined_labels + cohort_df["segment"].tolist()

# ───────────────────── PCA Projection (Map) ───────────────────
pca = make_pca(C, combined)
Z_centroids = project_2d(pca, C)
Z_buyers = project_2d(pca, buyers)
Z_extra = project_2d(pca, cohort_df[FEAT_IDS].values) if not cohort_df.empty else np.empty((0,2))

# ────────────────────────── Tabs ──────────────────────────────
tab_map, tab_people, tab_explore, tab_library, tab_data = st.tabs(
    ["Segmentation Map", "Individuals", "Buyer Explorer", "Offers & Messaging", "Data & Export"]
)

# ───────────────────── Segmentation Map ───────────────────────
with tab_map:
    st.subheader("Where buyers fall (2D projection of 10-feature space)")

    df_map = pd.DataFrame({
        "x": Z_buyers[:,0], "y": Z_buyers[:,1],
        "segment": labels, "type": ["Synthetic"]*len(labels)
    })
    if not cohort_df.empty:
        df_extra = pd.DataFrame({
            "x": Z_extra[:,0], "y": Z_extra[:,1],
            "segment": cohort_df["segment"], "type": ["Added"]*len(Z_extra)
        })
        df_map = pd.concat([df_map, df_extra], ignore_index=True)

    fig = px.scatter(
        df_map, x="x", y="y", color="segment", symbol="type",
        opacity=0.9, labels={"x":"Axis 1", "y":"Axis 2"},
        color_discrete_map=segment_palette(), height=620,
        hover_data={"segment":True, "x":":.2f","y":":.2f","type":True},
    )

    # segment centroids as anchors
    df_c = pd.DataFrame({
        "x": Z_centroids[:,0], "y": Z_centroids[:,1],
        "segment": [s["name"] for s in SEGMENTS]
    })
    fig.add_trace(go.Scatter(
        x=df_c["x"], y=df_c["y"],
        mode="markers+text",
        text=[n.split()[0] for n in df_c["segment"]],
        textposition="top center",
        marker=dict(size=18, line=dict(width=2, color="white"), symbol="diamond"),
        name="Segment centroid",
        hoverinfo="skip",
        showlegend=True
    ))

    # optional density overlay
    if show_density:
        fig.add_trace(go.Histogram2dContour(
            x=df_map["x"], y=df_map["y"],
            ncontours=6, showscale=False, colorscale="Greys", opacity=0.20,
            name="Density"
        ))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=30, t=20, b=60),
        legend=dict(orientation="h", y=1.05, x=0)
    )
    fig.add_hline(0, line_color="rgba(0,0,0,.15)")
    fig.add_vline(0, line_color="rgba(0,0,0,.15)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption("Tip: Add buyers in the sidebar to see them appear as diamonds overlayed with segment centroids.")

# ─────────────────────── Individuals ──────────────────────────
with tab_people:
    st.subheader("Describe the individuals → see where they land")
    # Build a compact card-like table for a small sample
    sample_idx = list(range(min(10, len(labels))))
    cards = []
    for i in sample_idx:
        # compute probabilities for the synthetic record
        x = buyers[i]
        probs, _ = nearest_centroid_probs(x, C, temp=temp)
        seg_idx = int(np.argmax(probs))
        seg_name = SEGMENTS[seg_idx]["name"]
        top2 = np.argsort(probs)[::-1][:2]
        cards.append(dict(
            Buyer=f"Buyer {i+1}",
            Segment=seg_name,
            Confidence=f"{(probs[top2[0]]*100):.0f}%",
            RunnerUp=SEGMENTS[top2[1]]["name"],
        ))
    st.dataframe(pd.DataFrame(cards), use_container_width=True)

    st.caption("This table summarizes a subset of buyers. Explore any buyer in detail in the next tab.")

# ─────────────────────── Buyer Explorer ───────────────────────
with tab_explore:
    st.subheader("Classify a single buyer & tailor the playbook")

    # choose source: synthetic or added
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
        probs, dists = nearest_centroid_probs(x, C, temp=temp)
        order = np.argsort(probs)[::-1]
        top = order[0]
        winner = SEGMENTS[top]

        # Show probability bars
        prob_df = pd.DataFrame({
            "Segment":[SEGMENTS[i]["name"] for i in order],
            "Probability":[float(probs[i]) for i in order]
        })
        bar = px.bar(prob_df, x="Probability", y="Segment", orientation="h",
                     range_x=[0,1], height=320, text=[f"{p*100:.0f}%" for p in prob_df["Probability"]],
                     color="Probability", color_continuous_scale="Blues")
        bar.update_layout(coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=80,r=30,t=20,b=40))
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar":False})

        # Radar vs centroid
        def _radar_row(name, vec):
            return {"Variable": name, **{k:v for k,v in zip(FEAT_IDS, vec)}}
        radar_df = pd.DataFrame([_radar_row("Buyer", x), _radar_row(winner["name"], vec_from_dict(winner["centroid"]))])
        rfig = px.line_polar(radar_df.melt(id_vars="Variable", var_name="Feature", value_name="Score"),
                             r="Score", theta="Feature", color="Variable", line_close=True, range_r=[0,5], height=420)
        rfig.update_traces(fill="toself")
        rfig.update_layout(showlegend=True, polar=dict(bgcolor="white"))
        st.plotly_chart(rfig, use_container_width=True, config={"displayModeBar":False})

        # Tailored guidance
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("#### 🧭 Recommended segment")
            st.metric(winner["name"], f"{probs[top]*100:.0f}% match")
        with c2:
            st.markdown("#### 💎 Value props to emphasize")
            for v in winner["value_props"]:
                st.write(f"- {v}")
        with c3:
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
    # build combined DF for export (synthetic only for brevity)
    syn_df = pd.DataFrame(buyers, columns=FEAT_IDS)
    # classify all for table
    segs = []
    for i in range(len(syn_df)):
        p,_ = nearest_centroid_probs(buyers[i], C, temp=temp)
        segs.append(SEGMENTS[int(np.argmax(p))]["name"])
    syn_df.insert(0, "segment", segs)
    syn_df.insert(0, "id", [f"buyer_{i+1}" for i in range(len(syn_df))])

    if not cohort_df.empty:
        export_df = pd.concat([syn_df, cohort_df.reset_index(drop=True)], ignore_index=True, sort=False)
    else:
        export_df = syn_df

    st.dataframe(export_df, use_container_width=True, height=380)
    st.download_button("⬇️ Download CSV", export_df.to_csv(index=False).encode(),
                       "graza_segmented_buyers.csv", "text/csv")
    st.caption("Columns are 10 features (0–5), plus inferred segment labels.")
