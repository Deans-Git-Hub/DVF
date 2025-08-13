






# graza_segments_app.py
"""
Graza Buyer Segmentation & Messaging — Demo
Axes: Axis 1 = Practicality, Axis 2 = Luxury (rotated PCA)
Run:
    pip install streamlit numpy pandas plotly scikit-learn
    streamlit run graza_segments_app.py
"""

import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from sklearn.decomposition import PCA
import streamlit as st

# ——— 1) Password gate ——————————————————————————
PASSWORD = st.secrets.get("password")
if PASSWORD is None:
    st.error(
        "⚠️ No `password` in secrets!\n\n"
        "Add in `.streamlit/secrets.toml`:\n\n"
        "    password = \"Synthetic!\"\n\n"
        "or set it in your Streamlit Cloud Secrets."
    )
    st.stop()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Please log in")
    with st.form("login_form"):
        pw     = st.text_input("Password", type="password", placeholder="••••••")
        submit = st.form_submit_button("Unlock")
    if submit:
        if pw == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password.")
    st.stop()







# ─────────────────────────── Setup ────────────────────────────
st.set_page_config(page_title="Graza Segmentation", layout="wide")
st.title("🫒 Graza Buyer Segmentation & Messaging (Demo)")

# Internal constants (no UI)
SEED = 1234          # RNG for reproducible synthetic data
SPREAD = 0.35        # cluster tightness around segment centroids
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

# ───────────────── Segments (re-evaluated centroids & GTM notes) ────────────────
SEGMENTS = [
    dict(
        id="seg_trendy",
        name="Trendy Hosts",
        # High Luxury, lower Practicality → top-left
        centroid={
            "flavor_adventure": 4.9, "health_clean": 4.2, "price_sensitivity": 1.2,
            "cook_freq": 3.6, "convenience_need": 3.0, "sustainability": 4.0,
            "squeeze_affinity": 4.3, "online_grocery": 3.6, "social_hosting": 5.0, "novelty_seeking": 5.0,
        },
        value_props=["Giftable design", "Wow factor for guests", "Seasonal collabs"],
        offers=["Limited edition drops", "Host bundle", "Creator collabs"],
        channels=["IG Reels / Pinterest", "Pop-ups & events", "PR / earned media"],
        messages=[
            "The bottle that starts conversations.",
            "Bring the wow to your next spread."
        ],
    ),
    dict(
        id="seg_foodie",
        name="Flavor-First Foodies",
        # High Luxury, moderate Practicality → top-left (but more practical than Trendy)
        centroid={
            "flavor_adventure": 4.6, "health_clean": 3.4, "price_sensitivity": 1.7,
            "cook_freq": 4.5, "convenience_need": 2.6, "sustainability": 3.3,
            "squeeze_affinity": 3.8, "online_grocery": 2.8, "social_hosting": 4.0, "novelty_seeking": 4.2,
        },
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
        # High Luxury, mid Practicality → top-center
        centroid={
            "flavor_adventure": 3.8, "health_clean": 4.9, "price_sensitivity": 2.0,
            "cook_freq": 3.9, "convenience_need": 2.6, "sustainability": 4.0,
            "squeeze_affinity": 3.4, "online_grocery": 3.2, "social_hosting": 2.6, "novelty_seeking": 2.8,
        },
        value_props=["Additive-free", "Freshness & traceability", "Transparent sourcing"],
        offers=["Trial size", "Subscribe & save (fresh rotation)", "Quality guarantee"],
        channels=["Health & wellness newsletters", "Retail endcaps w/ clean labels", "Search"],
        messages=[
            "Nothing added. Just fresh, traceable olive oil.",
            "Clean ingredients you can squeeze into every meal."
        ],
    ),
    dict(
        id="seg_eco",
        name="Sustainable Shoppers",
        # High Luxury, mid-to-higher Practicality → top-mid/right
        centroid={
            "flavor_adventure": 3.5, "health_clean": 4.1, "price_sensitivity": 2.6,
            "cook_freq": 3.5, "convenience_need": 3.1, "sustainability": 4.8,
            "squeeze_affinity": 3.7, "online_grocery": 3.2, "social_hosting": 3.0, "novelty_seeking": 3.2,
        },
        value_props=["Lower footprint packaging", "Responsible sourcing", "Durable bottle"],
        offers=["Refill / bulk formats", "Carbon-aware shipping", "Cause tie-ins"],
        channels=["Email + blog education", "Sustainability influencers", "Retail shelf talkers"],
        messages=[
            "Sustainably sourced, thoughtfully packaged.",
            "Better olive oil, better impact."
        ],
    ),
    dict(
        id="seg_value",
        name="Value-Seeking Families",
        # Low Luxury, high Practicality → bottom-right
        centroid={
            "flavor_adventure": 2.4, "health_clean": 2.8, "price_sensitivity": 5.0,
            "cook_freq": 3.4, "convenience_need": 3.8, "sustainability": 2.3,
            "squeeze_affinity": 4.3, "online_grocery": 3.0, "social_hosting": 1.9, "novelty_seeking": 1.8,
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
        # Low Luxury, highest Practicality → bottom-right (farthest right)
        centroid={
            "flavor_adventure": 3.4, "health_clean": 3.2, "price_sensitivity": 3.6,
            "cook_freq": 2.6, "convenience_need": 4.8, "sustainability": 2.8,
            "squeeze_affinity": 4.7, "online_grocery": 4.6, "social_hosting": 2.8, "novelty_seeking": 3.2,
        },
        value_props=["Squeeze-and-go convenience", "No-mess prep", "Consistent results"],
        offers=["Quick-start recipes", "Refill 2-pack", "Subscribe & save"],
        channels=["Short-form video", "Retail quick meal bays", "Food kits & DTC"],
        messages=[
            "Real flavor, zero fuss—squeeze and cook.",
            "Upgrade weeknights in one squeeze."
        ],
    ),
]
SEG_NAMES = [s["name"] for s in SEGMENTS]

def segment_palette():
    return {s["name"]: c for s, c in zip(SEGMENTS, px.colors.qualitative.Set2)}

# ───────────────────── Utility Functions ──────────────────────
def vec_from_dict(d): return np.array([float(d[k]) for k in FEAT_IDS])

def nearest_centroid_probs(x, centroids, temp=TEMP):
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
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)  # normalize for pretty plotting
    return Z

# Composites to align axes (Practicality & Luxury) inside the PCA plane
def practicality_score(X):  # X: (n, n_features)
    df = pd.DataFrame(X, columns=FEAT_IDS)
    return (0.50*df["convenience_need"] + 0.25*df["squeeze_affinity"]
            + 0.20*df["online_grocery"] + 0.05*df["cook_freq"]).values

def luxury_score(X):
    df = pd.DataFrame(X, columns=FEAT_IDS)
    ps_inv = 5.0 - df["price_sensitivity"]
    return (0.25*ps_inv + 0.25*df["flavor_adventure"] + 0.20*df["novelty_seeking"]
            + 0.15*df["social_hosting"] + 0.10*df["health_clean"] + 0.05*df["sustainability"]).values

# Helper: rotate 2D PCA coords so X ≈ Practicality and Y ≈ Luxury
def rotate_to_practicality_luxury(Z, X_features):
    # Targets (z-scored)
    P = practicality_score(X_features); P = (P - P.mean()) / (P.std() + 1e-9)
    L = luxury_score(X_features);      L = (L - L.mean()) / (L.std() + 1e-9)

    # Direction for Luxury in PCA plane
    coef_L, *_ = np.linalg.lstsq(Z, L, rcond=None)    # 2 weights for [PC1, PC2]
    vL = coef_L / (np.linalg.norm(coef_L) + 1e-9)

    # Direction for Practicality, orthogonalized vs Luxury
    coef_P, *_ = np.linalg.lstsq(Z, P, rcond=None)
    vP_raw = coef_P
    vP = vP_raw - (vP_raw @ vL) * vL
    vP = vP / (np.linalg.norm(vP) + 1e-9)

    # Rotation matrix whose columns are [vP, vL]
    R = np.stack([vP, vL], axis=1)  # shape (2,2)
    Z_rot = Z @ R
    return Z_rot, R

def axis_loadings_table_from_components(components):
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
buyers, labels = [], []
for _ in range(n_syn):
    sidx = rng.integers(0, len(SEGMENTS))
    base = C[sidx]
    sample = np.clip(rng.normal(base, SPREAD, size=len(FEAT_IDS)), 0, 5)
    buyers.append(sample)
    labels.append(SEGMENTS[sidx]["name"])
buyers = np.array(buyers)

# ───────────────────── PCA Projection + Rotation ──────────────
pca = make_pca(C, buyers)  # fit on centroids + buyers
Z_centroids = project_2d(pca, C)
Z_buyers = project_2d(pca, buyers)

# Rotate plane to align axes with our composites
X_all = np.vstack([buyers, C])
Z_all = np.vstack([Z_buyers, Z_centroids])
Z_all_rot, R = rotate_to_practicality_luxury(Z_all, X_all)

# Split back out
Z_buyers_rot    = Z_all_rot[:len(Z_buyers)]
Z_centroids_rot = Z_all_rot[len(Z_buyers):]

# Also rotate component loadings for the explainer table
components = pca.components_[:2].copy()              # (2, n_features), rows=PC1,PC2
effective_components = (R.T @ components)           # orientation after rotation

# ────────────────────────── Tabs ──────────────────────────────
tab_map, tab_people, tab_explore, tab_library, tab_data = st.tabs(
    ["Segmentation Map", "Individuals", "Buyer Explorer", "Offers & Messaging", "Data & Export"]
)

# ───────────────────── Segmentation Map ───────────────────────
with tab_map:
    st.subheader("Where buyers fall (Practicality × Luxury)")

    df_map = pd.DataFrame({
        "x": Z_buyers_rot[:,0], "y": Z_buyers_rot[:,1],
        "segment": labels, "type": ["Synthetic"]*len(labels)
    })

    # segment centroids as anchors
    df_c = pd.DataFrame({
        "x": Z_centroids_rot[:,0], "y": Z_centroids_rot[:,1],
        "segment": [s["name"] for s in SEGMENTS]
    })

    fig = px.scatter(
        df_map, x="x", y="y", color="segment", symbol="type",
        opacity=0.9,
        labels={"x": "Axis 1: Practicality", "y": "Axis 2: Luxury"},
        color_discrete_map=segment_palette(), height=620,
        hover_data={"segment":True,"x":":.2f","y":":.2f","type":True},
    )

    fig.add_trace(go.Scatter(
        x=df_c["x"], y=df_c["y"],
        mode="markers+text",
        text=[n.split()[0] for n in df_c["segment"]],
        textposition="top center",
        marker=dict(size=18, line=dict(width=2, color="white"), symbol="diamond"),
        name="Segment centroid", hoverinfo="skip", showlegend=True
    ))

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

    with st.expander("What drives these axes?"):
        st.write("Axes are PCA-projected and then rotated so X aligns with **Practicality** and Y with **Luxury** for this dataset.")
        st.dataframe(axis_loadings_table_from_components(effective_components), use_container_width=True)
        st.caption("Luxury blends lower price sensitivity with flavor/novelty/hosting and clean/sustainable cues.")

# ─────────────────────── Individuals ──────────────────────────
with tab_people:
    st.subheader("Backstory examples by segment")

    backstories = {
        "Trendy Hosts": [
            "Luca is the friend whose dinner parties end up on Stories. He curates tablescapes, gifts good-looking pantry staples, and hunts seasonal drops that spark conversation. Price is secondary to design, flavor ‘wow,’ and the joy of unveiling something new to guests—exactly where Graza’s limited editions and host bundles shine. Creator collabs and pop-up moments keep him engaged."
        ],
        "Flavor-First Foodies": [
            "Marisol is a Brooklyn home cook who hosts friends almost every weekend. She follows chef creators on Instagram and treats her pantry like a studio—always chasing a peppery finish or a new drizzle that wakes up roasted vegetables. Packaging matters, but she’ll pay more for single-origin flavor and chef-taught techniques she can execute on a weeknight. Recipe drops and the Drizzle + Sizzle bundle are her love language."
        ],
        "Clean-Label Purists": [
            "Jason is a wellness-focused parent who reads every label and compares sourcing claims. He wants fresh harvest, traceability, and a routine that keeps quality high—think subscribe-and-save with clear provenance. His decision unlocks when he sees certifications, storage guidance, and proof that Graza’s squeeze keeps oxygen out between uses. A quality guarantee removes any perceived risk."
        ],
        "Sustainable Shoppers": [
            "Ava prioritizes lower-footprint choices and responsible sourcing. She’s willing to pay a modest premium if she understands the environmental impact—refill options, durable bottles, and a transparent supply chain matter. She shares sustainability wins in group chats and appreciates brands that report progress, not perfection. A carbon-aware shipping note is a plus."
        ],
        "Value-Seeking Families": [
            "Tanya and Rob juggle two school schedules and weeknight dinners. They love that the squeeze bottle keeps counters clean and portions consistent, but they’re price-watchers first. Multi-packs, loyalty credits, and straightforward ‘feed-the-family’ recipes matter more than limited collabs or gifting stories. If it stretches the budget and reduces mess, it wins."
        ],
        "Time-Saving Meal Hackers": [
            "Dev is a startup PM who buys groceries online and cooks in 20-minute windows between calls. He wants reliable high-heat behavior and one-squeeze consistency, plus quick-start recipes that remove cognitive load. Trial bundles, refill two-packs, and short tutorial clips convert him quickly. He’ll happily subscribe once he trusts the routine."
        ],
    }

    for seg_name, stories in backstories.items():
        with st.expander(seg_name):
            for s_text in stories:
                st.write(s_text)

# ─────────────────────── Buyer Explorer ───────────────────────
with tab_explore:
    st.subheader("Classify a single buyer & tailor the playbook")

    idx = st.slider("Pick synthetic buyer", 1, len(buyers), 1)
    x = buyers[idx-1]

    probs, dists = nearest_centroid_probs(x, np.vstack([vec_from_dict(s["centroid"]) for s in SEGMENTS]), temp=TEMP)
    order = np.argsort(probs)[::-1]
    top = order[0]
    winner = SEGMENTS[top]

    # Probability bars
    prob_df = pd.DataFrame({
        "Segment":[SEGMENTS[i]["name"] for i in order],
        "Probability":[float(probs[i]) for i in order]
    })
    bar = px.bar(prob_df, x="Probability", y="Segment", orientation="h",
                 range_x=[0,1], height=320, text=[f"{p*100:.0f}%" for p in prob_df["Probability"]],
                 color="Probability", color_continuous_scale="Blues")
    bar.update_layout(coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=80,r=30,t=20,b=40))
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
    syn_df = pd.DataFrame(buyers, columns=FEAT_IDS)
    # classify all for table
    C_mat = np.vstack([vec_from_dict(s["centroid"]) for s in SEGMENTS])
    segs = []
    for i in range(len(syn_df)):
        p,_ = nearest_centroid_probs(buyers[i], C_mat, temp=TEMP)
        segs.append(SEGMENTS[int(np.argmax(p))]["name"])
    syn_df.insert(0, "segment", segs)
    syn_df.insert(0, "id", [f"buyer_{i+1}" for i in range(len(syn_df))])

    st.dataframe(syn_df, use_container_width=True, height=380)
    st.download_button("⬇️ Download CSV", syn_df.to_csv(index=False).encode(),
                       "graza_segmented_buyers.csv", "text/csv")
    st.caption("Columns are 10 features (0–5), plus inferred segment labels.")
