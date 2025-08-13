"""
Graza Buyer Segmentation & Messaging — Demo
Run:
    pip install streamlit numpy pandas plotly scikit-learn
    streamlit run graza_segments_app.py
"""

import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from sklearn.decomposition import PCA
import streamlit as st

# ─────────────────────────── Setup ────────────────────────────
st.set_page_config(page_title="Graza Segmentation", layout="wide")
st.title("🫒 Graza Buyer Segmentation & Messaging (Demo)")

# Internal constants (no UI)
SEED = 1234          # RNG for reproducible synthetic data
SPREAD = 0.35        # how tight buyers cluster around segment centroids
TEMP = 1.2           # segmentation temperature for softmax
SHOW_DENSITY = True  # show density contours on the map

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

# ───────────────── Segments (centroids + GTM guidance) ─────────────────
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
        # ↓ Tweaked to sit LOWER on Luxury (more value-seeking, less novelty/hosting/clean)
        centroid={
            "flavor_adventure": 2.5,
            "health_clean":     3.2,
            "price_sensitivity":4.9,
            "cook_freq":        3.4,
            "convenience_need": 3.4,
            "sustainability":   1.6,
            "squeeze_affinity": 2.1,
            "online_grocery":   3.2,
            "social_hosting":   1.4,
            "novelty_seeking":  2.1,
        },
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
        # ↑ Tweaked to sit HIGHER on Luxury (lower price sensitivity, more novelty/hosting/clean)
        centroid={
            "flavor_adventure": 4.7,
            "health_clean":     3.9,
            "price_sensitivity":1.7,
            "cook_freq":        3.7,
            "convenience_need": 2.1,
            "sustainability":   3.8,
            "squeeze_affinity": 4.2,
            "online_grocery":   2.6,
            "social_hosting":   4.9,
            "novelty_seeking":  4.9,
        },
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

def segment_palette():
    return {s["name"]: c for s, c in zip(SEGMENTS, px.colors.qualitative.Set2)}

# ───────────────────── Utility Functions ──────────────────────
def vec_from_dict(d):
    return np.array([float(d[k]) for k in FEAT_IDS])

def nearest_centroid_probs(x, centroids, temp=TEMP):
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

# PCA components helper
def _pca_components(p):
    return p.components_ if hasattr(p, "components_") else p["components"]

def axis_loadings_table_from_components(components):
    """Show top positive/negative features for each axis (after orientation)."""
    feat_names = dict(FEATURES)
    rows = []
    for i, w in enumerate(components, start=1):
        s = pd.Series(w, index=FEAT_IDS).sort_values()
        rows.append({
            "Axis": f"Axis {i}",
            "Top (+)": ", ".join(feat_names[f] for f in s.tail(3).index),
            "Top (–)": ", ".join(feat_names[f] for f in s.head(3).index),
        })
    return pd.DataFrame(rows)

# ─────────────────────── Sidebar Inputs ───────────────────────
with st.sidebar:
    st.header("Controls")
    n_syn = st.slider("Synthetic buyers", 30, 400, 120, 10)

# ─────────────────── Synthetic Buyer Cohort ───────────────────
rng = np.random.default_rng(SEED)

# build centroid matrix
C = np.vstack([vec_from_dict(s["centroid"]) for s in SEGMENTS])

# sample synthetic buyers around centroids
buyers = []
labels = []
for _ in range(n_syn):
    sidx = rng.integers(0, len(SEGMENTS))  # pick a segment
    base = C[sidx]
    sample = rng.normal(base, SPREAD, size=len(FEAT_IDS))
    sample = np.clip(sample, 0, 5)
    buyers.append(sample)
    labels.append(SEGMENTS[sidx]["name"])
buyers = np.array(buyers)

# ───────────────────── PCA Projection (Map) ───────────────────
pca = make_pca(C, buyers)  # fit on centroids + current buyers
Z_centroids = project_2d(pca, C)
Z_buyers = project_2d(pca, buyers)

# Enforce Axis 2 orientation: Luxury ↑ / Value-seeking ↓
components = _pca_components(pca)[:2].copy()
comp1 = components[0].copy()
comp2 = components[1].copy()
ps_idx = FEAT_IDS.index("price_sensitivity")
flip2 = comp2[ps_idx] > 0  # if PC2 increases with price_sensitivity, flip it
if flip2:
    Z_buyers[:, 1] *= -1
    Z_centroids[:, 1] *= -1
    comp2 *= -1
effective_components = [comp1, comp2]  # for the explainer table

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

    fig = px.scatter(
        df_map, x="x", y="y", color="segment", symbol="type",
        opacity=0.9,
        labels={"x": "Axis 1: Practicality", "y": "Axis 2: Luxury"},
        color_discrete_map=segment_palette(), height=620,
        hover_data={"segment":True,"x":":.2f","y":":.2f","type":True},
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

    # optional density overlay (always on in this simplified version)
    if SHOW_DENSITY:
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

    with st.expander("What do Axis 1 and Axis 2 mean here?"):
        st.write("They’re PCA directions (weighted blends of features) fitted on this dataset.")
        st.dataframe(axis_loadings_table_from_components(effective_components), use_container_width=True)
        st.caption("Axis 2 is oriented so **up = more Luxury (lower price sensitivity)**, **down = more value-seeking**.")

# ─────────────────────── Individuals ──────────────────────────
with tab_people:
    st.subheader("Backstory examples by segment")

    backstories = {
        "Flavor-First Foodies": [
            "Marisol is a Brooklyn home cook who hosts friends almost every weekend. She follows chef creators on Instagram and treats her pantry like a studio—always chasing a peppery finish or a new drizzle that wakes up roasted vegetables. Packaging matters, but she’ll pay more for single-origin flavor and chef-taught techniques she can execute on a weeknight."
        ],
        "Clean-Label Purists": [
            "Jason is a wellness-focused parent who reads every label and compares sourcing claims. He wants fresh harvest, traceability, and a routine that keeps quality high—think subscribe-and-save with clear provenance. His decision unlocks when he sees certifications, storage guidance, and proof that Graza’s squeeze keeps oxygen out between uses."
        ],
        "Value-Seeking Families": [
            "Tanya and Rob juggle two school schedules and weeknight dinners. They love that the squeeze bottle keeps counters clean and portions consistent, but they’re price-watchers first. Multi-packs, loyalty credits, and simple ‘feed-the-family’ recipes speak to them far more than limited collabs or gifting stories."
        ],
        "Time-Saving Meal Hackers": [
            "Dev is a startup PM who buys groceries online and cooks in 20-minute windows between calls. He wants reliable heat behavior and one-squeeze consistency, plus quick-start recipes that remove cognitive load. Trial bundles, refill two-packs, and short tutorial clips convert him quickly."
        ],
        "Sustainable Shoppers": [
            "Ava prioritizes lower-footprint choices and responsible sourcing. She’s willing to pay a modest premium if she understands the environmental impact—refill options, durable bottles, and a transparent supply chain matter. She shares sustainability wins in her group chats and appreciates brands that report progress, not perfection."
        ],
        "Trendy Hosts": [
            "Luca is the friend whose dinner parties end up on Stories. He curates tablescapes, gifts good-looking pantry staples, and loves seasonal drops that spark conversation. Price is a minor consideration compared with design, flavor ‘wow,’ and the joy of unveiling something new to guests—exactly where Graza’s limited editions and host bundles shine."
        ],
    }

    for seg_name, stories in backstories.items():
        with st.expander(seg_name):
            for s_text in stories:
                st.write(s_text)

# ─────────────────────── Buyer Explorer ───────────────────────
with tab_explore:
    st.subheader("Classify a single buyer & tailor the playbook")

    # Only synthetic buyers in this simplified version
    idx = st.slider("Pick synthetic buyer", 1, len(labels), 1)
    x = buyers[idx-1]

    probs, dists = nearest_centroid_probs(x, C, temp=TEMP)
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
    # build synthetic DF for export
    syn_df = pd.DataFrame(buyers, columns=FEAT_IDS)
    # classify all for table
    segs = []
    for i in range(len(syn_df)):
        p,_ = nearest_centroid_probs(buyers[i], C, temp=TEMP)
        segs.append(SEGMENTS[int(np.argmax(p))]["name"])
    syn_df.insert(0, "segment", segs)
    syn_df.insert(0, "id", [f"buyer_{i+1}" for i in range(len(syn_df))])

    st.dataframe(syn_df, use_container_width=True, height=380)
    st.download_button("⬇️ Download CSV", syn_df.to_csv(index=False).encode(),
                       "graza_segmented_buyers.csv", "text/csv")
    st.caption("Columns are 10 features (0–5), plus inferred segment labels.")
