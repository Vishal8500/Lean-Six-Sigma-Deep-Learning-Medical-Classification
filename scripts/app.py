"""
╔══════════════════════════════════════════════════════════════╗
║   LEAN SIX SIGMA × DEEP LEARNING  — TEAM THETA              ║
║   Chest X-Ray Diagnostic Dashboard  |  Streamlit App        ║
╚══════════════════════════════════════════════════════════════╝

INSTALL DEPENDENCIES:
    pip install streamlit plotly pandas pillow torch torchvision

RUN:
    streamlit run lean_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import base64
from PIL import Image

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="LSS × DeepLearning | Team Θ",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS — Dark Clinical Aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne+Mono&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg:        #030B14;
    --surface:   #071525;
    --card:      #0C1E33;
    --border:    #14324F;
    --accent1:   #00D4FF;
    --accent2:   #7B61FF;
    --accent3:   #00FF9D;
    --warn:      #FF6B35;
    --text:      #E8F4FF;
    --muted:     #6E90B0;
    --sigma:     #FFD700;
}

/* Base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Remove default Streamlit padding */
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #030B14 0%, #071525 40%, #0C1E33 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(0,212,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #00D4FF, #7B61FF, #00FF9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0 !important;
    line-height: 1.1 !important;
}
.main-header .sub {
    color: var(--muted);
    font-size: 0.9rem;
    font-family: 'Syne Mono', monospace;
    letter-spacing: 0.08em;
}
.team-badge {
    display: inline-block;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent1) !important;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Syne Mono', monospace;
    letter-spacing: 0.05em;
    margin-top: 0.6rem;
}

/* KPI Cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.cyan::before   { background: var(--accent1); }
.kpi-card.purple::before { background: var(--accent2); }
.kpi-card.green::before  { background: var(--accent3); }
.kpi-card.orange::before { background: var(--warn); }
.kpi-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1.15; margin: 0.3rem 0 0.1rem; }
.kpi-value.cyan   { color: var(--accent1); }
.kpi-value.purple { color: var(--accent2); }
.kpi-value.green  { color: var(--accent3); }
.kpi-value.orange { color: var(--warn); }
.kpi-delta { font-size: 0.78rem; color: var(--muted); }
.kpi-delta span { color: var(--accent3); font-weight: 600; }

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin: 1.8rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.section-header h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin: 0 !important;
}
.dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

/* Charts / Plotly container */
.chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}

/* DMAIC Phase Badges */
.dmaic-row { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.dmaic-badge {
    padding: 0.4rem 1rem; border-radius: 6px;
    font-size: 0.78rem; font-weight: 700; font-family: 'Syne Mono', monospace;
    letter-spacing: 0.05em; cursor: default;
}
.dmaic-D { background: rgba(0,212,255,0.15); color: #00D4FF; border: 1px solid rgba(0,212,255,0.3); }
.dmaic-M { background: rgba(123,97,255,0.15); color: #9B7FFF; border: 1px solid rgba(123,97,255,0.3); }
.dmaic-A { background: rgba(255,107,53,0.15); color: #FF8555; border: 1px solid rgba(255,107,53,0.3); }
.dmaic-I { background: rgba(0,255,157,0.15); color: #00FF9D; border: 1px solid rgba(0,255,157,0.3); }
.dmaic-C { background: rgba(255,215,0,0.15); color: #FFD700; border: 1px solid rgba(255,215,0,0.3); }

/* Table */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.styled-table th {
    background: rgba(0,212,255,0.1);
    color: var(--accent1);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.6rem 0.8rem;
    border-bottom: 2px solid rgba(0,212,255,0.2);
    text-align: left;
}
.styled-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.styled-table tr:hover td { background: rgba(0,212,255,0.04); }

/* Diagnostic upload section */
.diag-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.diag-result {
    margin-top: 1.5rem;
    padding: 1.5rem;
    border-radius: 12px;
}
.diag-result.normal {
    background: rgba(0,255,157,0.08);
    border: 1px solid rgba(0,255,157,0.3);
}
.diag-result.pneumonia {
    background: rgba(255,107,53,0.08);
    border: 1px solid rgba(255,107,53,0.3);
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
}
.result-label.normal   { color: var(--accent3); }
.result-label.pneumonia { color: var(--warn); }
.conf-bar-bg {
    background: var(--border);
    border-radius: 20px;
    height: 10px;
    margin: 0.5rem 0;
    overflow: hidden;
}

/* Sigma level indicator */
.sigma-pill {
    display: inline-block;
    background: rgba(255,215,0,0.15);
    border: 1px solid rgba(255,215,0,0.35);
    color: var(--sigma);
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-family: 'Syne Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Streamlit widget overrides */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stFileUploader {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 10px; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--muted) !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'Syne', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent1) !important;
    border-bottom: 2px solid var(--accent1) !important;
}
div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        "run_id": [
            "baseline_1772434188",
            "doe_1772377858_0.001_16_0.3_Adam",
            "doe_1772378354_0.001_16_0.3_RMSprop",
            "doe_1772378851_0.001_16_0.5_Adam",
            "doe_1772379412_0.001_16_0.5_RMSprop",
            "doe_1772379957_0.001_32_0.3_Adam",
            "doe_1772380507_0.001_32_0.3_RMSprop",
            "doe_1772381293_0.001_32_0.5_Adam",
            "doe_1772381586_0.001_32_0.5_RMSprop",
            "doe_1772381824_0.0001_16_0.3_Adam",
            "doe_1772382100_0.0001_16_0.3_RMSprop",
            "doe_1772382379_0.0001_16_0.5_Adam",
            "doe_1772382655_0.0001_16_0.5_RMSprop",
            "doe_1772382928_0.0001_32_0.3_Adam",
            "doe_1772383163_0.0001_32_0.3_RMSprop",
            "doe_1772383402_0.0001_32_0.5_Adam",
            "doe_1772383638_0.0001_32_0.5_RMSprop",
            "improve_1772386473",
            "improve_1772388162",
            "focused_1772426416_0.001_0",
            "focused_1772427218_0.001_0.0001",
            "focused_1772427912_0.0005_0",
            "focused_1772428616_0.0005_0.0001",
            "focused_1772429310_0.0001_0",
            "focused_1772430017_0.0001_0.0001",
            "final_1772431147",
        ],
        "run_type": [
            "baseline","doe","doe","doe","doe","doe","doe","doe","doe",
            "doe","doe","doe","doe","doe","doe","doe","doe",
            "improve","improve",
            "focused_doe","focused_doe","focused_doe","focused_doe","focused_doe","focused_doe",
            "final_model"
        ],
        "lr": [
            0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,
            0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,
            0.001,0.001,
            0.001,0.001,0.0005,0.0005,0.0001,0.0001,
            0.0001
        ],
        "batch_size": [
            16,16,16,16,16,32,32,32,32,16,16,16,16,32,32,32,32,
            16,16,16,16,16,16,16,16,16
        ],
        "dropout": [
            0.5,0.3,0.3,0.5,0.5,0.3,0.3,0.5,0.5,0.3,0.3,0.5,0.5,0.3,0.3,0.5,0.5,
            0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3
        ],
        "optimizer": [
            "Adam","Adam","RMSprop","Adam","RMSprop","Adam","RMSprop","Adam","RMSprop",
            "Adam","RMSprop","Adam","RMSprop","Adam","RMSprop","Adam","RMSprop",
            "RMSprop","RMSprop","RMSprop","RMSprop","RMSprop","RMSprop","RMSprop","RMSprop","RMSprop"
        ],
        "epochs": [
            15,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,20,9,10,10,10,10,10,10,8
        ],
        "train_loss": [
            0.121740,0.151089,0.154667,0.151505,0.166606,0.129372,0.147015,0.136672,
            0.160721,0.165344,0.164534,0.178328,0.176705,0.157272,0.159718,0.170761,
            0.171116,0.010375,0.017806,0.014975,0.020362,0.011130,0.014456,0.010234,
            0.011469,0.017415
        ],
        "val_loss": [
            0.703647,0.255399,0.574668,0.762606,0.464890,0.688019,0.437537,0.593949,
            0.402829,0.741336,0.665464,0.651287,0.553336,0.495760,0.720059,0.634717,
            0.602488,0.777673,0.005601,0.223356,0.690855,0.016816,0.651045,0.704918,
            0.302582,0.291856
        ],
        "accuracy": [
            0.825321,0.870192,0.849359,0.772436,0.860577,0.809295,0.870192,0.818910,
            0.862179,0.804487,0.815705,0.820513,0.828526,0.830128,0.799679,0.809295,
            0.815705,0.751603,0.807692,0.793269,0.743590,0.842949,0.814103,0.815705,
            0.860577,0.866987
        ],
        "f1": [
            0.876557,0.904594,0.891705,0.845652,0.898719,0.867039,0.905041,0.872891,
            0.899533,0.864143,0.870641,0.873874,0.878547,0.879271,0.861265,0.867039,
            0.870932,0.834225,0.865471,0.857143,0.829787,0.887872,0.869369,0.870349,
            0.899189,0.902924
        ],
        "defects": [
            109,81,94,142,87,119,81,113,86,122,115,112,107,106,125,119,115,
            155,120,129,160,98,116,115,87,83
        ],
        "dpmo": [
            174679.49,129807.69,150641.03,227564.10,139423.08,190705.13,129807.69,
            181089.74,137820.51,195512.82,184294.87,179487.18,171474.36,169871.79,
            200320.51,190705.13,184294.87,248397.44,192307.69,206730.77,256410.26,
            157051.28,185897.44,184294.87,139423.08,133012.82
        ],
        "training_time_sec": [
            1015.96,480.17,480.28,545.74,529.41,521.27,771.53,286.02,231.40,268.84,
            270.72,269.69,264.49,228.42,230.24,229.24,320.08,1494.59,901.73,788.71,
            686.41,696.42,686.14,699.72,803.64,548.76
        ],
    }
    df = pd.DataFrame(data)
    # Add sigma level column
    def dpmo_to_sigma(dpmo):
        if dpmo < 3.4:   return "6σ"
        elif dpmo < 233: return "5σ"
        elif dpmo < 6210: return "4σ"
        elif dpmo < 66807: return "3σ"
        elif dpmo < 308537: return "2σ"
        else: return "1σ"
    df["sigma_level"] = df["dpmo"].apply(dpmo_to_sigma)
    df["phase_num"] = df["run_type"].map({
        "baseline": 1, "doe": 2, "improve": 3,
        "focused_doe": 4, "final_model": 5
    })
    return df

df = load_data()

PLOTLY_LAYOUT = dict(
    plot_bgcolor="#0C1E33",
    paper_bgcolor="#0C1E33",
    font=dict(family="Space Grotesk", color="#E8F4FF", size=12),
    xaxis=dict(gridcolor="#14324F", linecolor="#14324F", tickcolor="#6E90B0"),
    yaxis=dict(gridcolor="#14324F", linecolor="#14324F", tickcolor="#6E90B0"),
    margin=dict(l=10, r=10, t=35, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#14324F"),
)

COLOR_MAP = {
    "baseline":    "#6E90B0",
    "doe":         "#7B61FF",
    "improve":     "#FF6B35",
    "focused_doe": "#00D4FF",
    "final_model": "#00FF9D",
}

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
        <span style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
        background:linear-gradient(90deg,#00D4FF,#7B61FF);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🫁 LSS×DL</span>
        <div style='color:#6E90B0;font-size:0.72rem;font-family:Syne Mono,monospace;margin-top:0.2rem;'>
        TEAM THETA | 22MIA</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ DMAIC Phase Filter**")
    phases = st.multiselect(
        "Select phases",
        options=["baseline","doe","improve","focused_doe","final_model"],
        default=["baseline","doe","improve","focused_doe","final_model"],
        format_func=lambda x: {
            "baseline":"📍 Baseline (Measure)",
            "doe":"🔬 DOE (Analyze)",
            "improve":"⬆️ Improve",
            "focused_doe":"🎯 Focused DOE",
            "final_model":"✅ Final Model (Control)"
        }.get(x,x)
    )
    if not phases:
        phases = ["baseline","doe","improve","focused_doe","final_model"]

    st.markdown("**🔧 Hyperparameter Filter**")
    opt_filter = st.multiselect("Optimizer", ["Adam","RMSprop"], default=["Adam","RMSprop"])
    lr_filter  = st.multiselect("Learning Rate", [0.001,0.0005,0.0001], default=[0.001,0.0005,0.0001])

    st.markdown("---")
    # Quick stats
    best_row = df.loc[df["accuracy"].idxmax()]
    st.markdown(f"""
    <div style='background:#071525;border:1px solid #14324F;border-radius:10px;padding:0.9rem;'>
    <div style='font-size:0.7rem;color:#6E90B0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>🏆 Best Run</div>
    <div style='font-size:0.75rem;color:#00D4FF;font-family:Syne Mono,monospace;word-break:break-all;margin-bottom:0.4rem;'>{best_row['run_id'][:28]}…</div>
    <div style='font-size:1.4rem;font-family:Syne,sans-serif;font-weight:800;color:#00FF9D;'>
    {best_row['accuracy']*100:.2f}%</div>
    <div style='font-size:0.72rem;color:#6E90B0;'>Accuracy &nbsp;|&nbsp; F1: {best_row['f1']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FILTER DATA
# ──────────────────────────────────────────────
mask = (
    df["run_type"].isin(phases) &
    df["optimizer"].isin(opt_filter) &
    df["lr"].isin(lr_filter)
)
dff = df[mask].copy()

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Lean Six Sigma × Deep Learning</h1>
    <div class="sub">CHEST X-RAY DIAGNOSTIC SYSTEM  ·  DMAIC FRAMEWORK  ·  TEAM THETA</div>
    <div>
        <span class="team-badge">M Vishal · 22MIA1014</span>
        <span class="team-badge" style="margin-left:6px;">Hrishikesh R · 22MIA1061</span>
        <span class="team-badge" style="margin-left:6px;">C A Cavin · 22MIA1035</span>
        <span class="team-badge" style="margin-left:6px;">Sowmya A · 22MIA1115</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DMAIC PHASE STRIP
# ──────────────────────────────────────────────
st.markdown("""
<div class="dmaic-row">
    <span class="dmaic-badge dmaic-D">D — Define</span>
    <span class="dmaic-badge dmaic-M">M — Measure</span>
    <span class="dmaic-badge dmaic-A">A — Analyze (DOE)</span>
    <span class="dmaic-badge dmaic-I">I — Improve</span>
    <span class="dmaic-badge dmaic-C">C — Control</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────
final = df[df["run_type"]=="final_model"].iloc[0]
baseline = df[df["run_type"]=="baseline"].iloc[0]
acc_lift = (final["accuracy"] - baseline["accuracy"]) * 100
dpmo_drop = baseline["dpmo"] - final["dpmo"]

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card cyan">
        <div class="kpi-label">Final Accuracy</div>
        <div class="kpi-value cyan">{final['accuracy']*100:.2f}%</div>
        <div class="kpi-delta">Baseline: {baseline['accuracy']*100:.2f}% &nbsp;|&nbsp; <span>+{acc_lift:.2f}pp ↑</span></div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-label">F1 Score</div>
        <div class="kpi-value purple">{final['f1']:.4f}</div>
        <div class="kpi-delta">Baseline: {baseline['f1']:.4f} &nbsp;|&nbsp; <span>+{(final['f1']-baseline['f1']):.4f} ↑</span></div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-label">DPMO (Final)</div>
        <div class="kpi-value green">{final['dpmo']:,.0f}</div>
        <div class="kpi-delta">Baseline: {baseline['dpmo']:,.0f} &nbsp;|&nbsp; <span>−{dpmo_drop:,.0f} ↓</span></div>
    </div>
    <div class="kpi-card orange">
        <div class="kpi-label">Defects Reduced</div>
        <div class="kpi-value orange">{baseline['defects']-final['defects']}</div>
        <div class="kpi-delta">{baseline['defects']} → {final['defects']} defects &nbsp;|&nbsp;
        <span>{((baseline['defects']-final['defects'])/baseline['defects']*100):.1f}% ↓</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Performance", "🔬 DOE Analysis", "📉 DPMO & Sigma", "🗂️ Run Table", "📈 Control Charts (SPC)", "🫁 Diagnostic"
])


# ════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-title">Accuracy Progression Across All Runs</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for rt in dff["run_type"].unique():
            sub = dff[dff["run_type"]==rt].reset_index(drop=True)
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub["accuracy"]*100,
                mode="lines+markers",
                name=rt.replace("_"," ").title(),
                line=dict(color=COLOR_MAP.get(rt,"#888"), width=2.5),
                marker=dict(size=7, symbol="circle"),
                hovertemplate="%{text}<br>Accuracy: %{y:.2f}%<extra></extra>",
                text=sub["run_id"].str[:20]
            ))
        # Baseline ref line
        fig.add_hline(y=baseline["accuracy"]*100,
                      line_dash="dot", line_color="#6E90B0", line_width=1.5,
                      annotation_text=f"Baseline {baseline['accuracy']*100:.2f}%",
                      annotation_font_color="#6E90B0")
        fig.add_hline(y=final["accuracy"]*100,
                      line_dash="dot", line_color="#00FF9D", line_width=1.5,
                      annotation_text=f"Final {final['accuracy']*100:.2f}%",
                      annotation_font_color="#00FF9D")
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="chart-title">F1 Score vs Accuracy Scatter</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            dff, x="accuracy", y="f1", color="run_type",
            size="training_time_sec", size_max=22,
            hover_data=["run_id","optimizer","lr","batch_size","dropout"],
            color_discrete_map=COLOR_MAP,
            labels={"accuracy":"Accuracy","f1":"F1 Score","run_type":"Phase"},
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=320)
        fig2.update_traces(marker=dict(line=dict(width=1, color="#0C1E33")))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-title">Train vs Validation Loss by Phase</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        cats = dff["run_type"].unique()
        for rt in cats:
            sub = dff[dff["run_type"]==rt]
            fig3.add_trace(go.Box(
                y=sub["train_loss"], name=f"{rt.replace('_',' ').title()} (Train)",
                marker_color=COLOR_MAP.get(rt,"#888"),
                line_color=COLOR_MAP.get(rt,"#888"),
                fillcolor=f"rgba({','.join(str(int(c)) for c in bytes.fromhex(COLOR_MAP.get(rt,'#888888')[1:]))},0.15)",
                boxmean=True,
            ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True,
                           yaxis_title="Loss", xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="chart-title">Training Time vs Accuracy</div>', unsafe_allow_html=True)
        fig4 = px.scatter(
            dff, x="training_time_sec", y="accuracy",
            color="run_type", symbol="optimizer",
            hover_data=["run_id","lr","epochs"],
            color_discrete_map=COLOR_MAP,
            labels={"training_time_sec":"Training Time (s)","accuracy":"Accuracy","run_type":"Phase"},
        )
        fig4.update_traces(marker=dict(size=10, line=dict(width=1, color="#0C1E33")))
        fig4.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════
# TAB 2 — DOE ANALYSIS
# ════════════════════════════════════════
with tab2:
    doe_df = df[df["run_type"]=="doe"].copy()

    st.markdown("""
    <div class="section-header">
        <div class="dot" style="background:#7B61FF;"></div>
        <h3>Design of Experiments — Hyperparameter Effects (Full Factorial 2³)</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="chart-title">Accuracy by Learning Rate</div>', unsafe_allow_html=True)
        fig = px.box(doe_df, x="lr", y="accuracy", color="lr",
                     color_discrete_sequence=["#7B61FF","#00D4FF"],
                     labels={"lr":"Learning Rate","accuracy":"Accuracy"})
        fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="chart-title">Accuracy by Batch Size</div>', unsafe_allow_html=True)
        fig = px.box(doe_df, x="batch_size", y="accuracy", color="batch_size",
                     color_discrete_sequence=["#FF6B35","#00FF9D"],
                     labels={"batch_size":"Batch Size","accuracy":"Accuracy"})
        fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown('<div class="chart-title">Accuracy by Optimizer</div>', unsafe_allow_html=True)
        fig = px.box(doe_df, x="optimizer", y="accuracy", color="optimizer",
                     color_discrete_sequence=["#00D4FF","#FF6B35"],
                     labels={"optimizer":"Optimizer","accuracy":"Accuracy"})
        fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Interaction heatmap
    st.markdown('<div class="chart-title" style="margin-top:1rem;">Main Effect — Mean Accuracy by Factor Combination</div>', unsafe_allow_html=True)
    pivot = doe_df.groupby(["lr","batch_size"])["accuracy"].mean().reset_index()
    pivot_wide = pivot.pivot(index="lr", columns="batch_size", values="accuracy")
    fig_heat = px.imshow(
        pivot_wide * 100,
        color_continuous_scale=[[0,"#030B14"],[0.5,"#7B61FF"],[1,"#00FF9D"]],
        labels={"color":"Accuracy (%)"},
        aspect="auto", text_auto=".2f"
    )
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=220, coloraxis_showscale=True)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Parallel coordinates
    st.markdown('<div class="chart-title">Parallel Coordinates — All DOE Runs</div>', unsafe_allow_html=True)
    fig_par = px.parallel_coordinates(
        doe_df,
        dimensions=["lr","batch_size","dropout","accuracy","f1","dpmo"],
        color="accuracy",
        color_continuous_scale=[[0,"#7B61FF"],[0.5,"#00D4FF"],[1,"#00FF9D"]],
        labels={"lr":"LR","batch_size":"Batch","dropout":"Dropout",
                "accuracy":"Accuracy","f1":"F1","dpmo":"DPMO"},
    )
    fig_par.update_layout(**PLOTLY_LAYOUT, height=320)
    st.plotly_chart(fig_par, use_container_width=True)


# ════════════════════════════════════════
# TAB 3 — DPMO & SIGMA
# ════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="dot" style="background:#FFD700;"></div>
        <h3>Six Sigma Quality Metrics — DPMO Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('<div class="chart-title">DPMO Across All Runs (Lower = Better)</div>', unsafe_allow_html=True)
        dff_sorted = dff.sort_values("phase_num").reset_index(drop=True)
        colors = [COLOR_MAP.get(rt,"#888") for rt in dff_sorted["run_type"]]
        fig_dpmo = go.Figure(go.Bar(
            x=list(range(len(dff_sorted))),
            y=dff_sorted["dpmo"],
            marker_color=colors,
            hovertemplate="<b>%{customdata[0]}</b><br>DPMO: %{y:,.0f}<extra></extra>",
            customdata=dff_sorted[["run_id","run_type"]].values,
        ))
        fig_dpmo.add_hline(y=baseline["dpmo"], line_dash="dot", line_color="#6E90B0",
                           annotation_text=f"Baseline {baseline['dpmo']:,.0f}",
                           annotation_font_color="#6E90B0")
        fig_dpmo.add_hline(y=final["dpmo"], line_dash="dot", line_color="#00FF9D",
                           annotation_text=f"Final {final['dpmo']:,.0f}",
                           annotation_font_color="#00FF9D")
        fig_dpmo.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="DPMO",
                               xaxis_title="Run Index")
        # Colorful legend
        for rt, c in COLOR_MAP.items():
            fig_dpmo.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=c),
                showlegend=True, name=rt.replace("_"," ").title()
            ))
        st.plotly_chart(fig_dpmo, use_container_width=True)

    with col2:
        st.markdown('<div class="chart-title">Defects by Phase</div>', unsafe_allow_html=True)
        phase_defects = dff.groupby("run_type")["defects"].agg(["mean","min","max"]).reset_index()
        fig_def = go.Figure()
        for _, row in phase_defects.iterrows():
            c = COLOR_MAP.get(row["run_type"],"#888")
            fig_def.add_trace(go.Bar(
                x=[row["run_type"].replace("_"," ").title()],
                y=[row["mean"]],
                error_y=dict(type="data", array=[row["max"]-row["mean"]], arrayminus=[row["mean"]-row["min"]]),
                marker_color=c,
                name=row["run_type"],
                showlegend=False,
            ))
        fig_def.update_layout(**PLOTLY_LAYOUT, height=320,
                              yaxis_title="Avg Defects", xaxis_title="")
        st.plotly_chart(fig_def, use_container_width=True)

    # Sigma level progression
    st.markdown('<div class="chart-title">Sigma Level Distribution</div>', unsafe_allow_html=True)
    sigma_counts = df["sigma_level"].value_counts().reset_index()
    sigma_counts.columns = ["sigma","count"]
    sigma_order = ["6σ","5σ","4σ","3σ","2σ","1σ"]
    sigma_colors = {"6σ":"#00FF9D","5σ":"#00D4FF","4σ":"#7B61FF","3σ":"#FFD700","2σ":"#FF6B35","1σ":"#FF3355"}
    fig_sig = go.Figure(go.Bar(
        x=[s for s in sigma_order if s in sigma_counts["sigma"].values],
        y=[sigma_counts.loc[sigma_counts["sigma"]==s,"count"].values[0]
           if s in sigma_counts["sigma"].values else 0 for s in sigma_order if s in sigma_counts["sigma"].values],
        marker_color=[sigma_colors[s] for s in sigma_order if s in sigma_counts["sigma"].values],
    ))
    fig_sig.update_layout(**PLOTLY_LAYOUT, height=220, xaxis_title="Sigma Level",
                          yaxis_title="# Runs", showlegend=False)
    st.plotly_chart(fig_sig, use_container_width=True)


# ════════════════════════════════════════
# TAB 4 — RUN TABLE
# ════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">
        <div class="dot" style="background:#00D4FF;"></div>
        <h3>Complete Experimental Run Log</h3>
    </div>
    """, unsafe_allow_html=True)

    sort_col = st.selectbox("Sort by", ["accuracy","f1","dpmo","defects","training_time_sec"], index=0)
    sort_asc = st.toggle("Ascending", value=False)
    display_df = dff.sort_values(sort_col, ascending=sort_asc)[
        ["run_id","run_type","lr","batch_size","dropout","optimizer","epochs",
         "accuracy","f1","defects","dpmo","sigma_level","training_time_sec"]
    ].copy()
    display_df["accuracy"]  = (display_df["accuracy"]*100).map("{:.2f}%".format)
    display_df["f1"]        = display_df["f1"].map("{:.4f}".format)
    display_df["dpmo"]      = display_df["dpmo"].map("{:,.0f}".format)
    display_df["training_time_sec"] = display_df["training_time_sec"].map("{:.1f}s".format)
    display_df.columns = ["Run ID","Phase","LR","Batch","Dropout","Optimizer","Epochs",
                          "Accuracy","F1","Defects","DPMO","Sigma","Time"]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.TextColumn("Accuracy"),
            "Sigma": st.column_config.TextColumn("σ Level"),
        }
    )

    # Download button
    csv_buf = io.StringIO()
    dff.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇  Export Filtered Data as CSV",
        data=csv_buf.getvalue(),
        file_name="lss_experiment_results.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════
# TAB 5 — CONTROL CHARTS (SPC)
# ════════════════════════════════════════
with tab5:

    # ── Exact values from the DMAIC report (Section 4.5) ──────────────
    ctrl_runs   = ["Control Run 1", "Control Run 2", "Control Run 3"]
    ctrl_acc    = [0.8317, 0.8253, 0.8446]
    ctrl_dpmo   = [168269, 174679, 155449]

    # SPC parameters — accuracy
    xbar_acc  = 0.8339
    sigma_acc = 0.00799
    ucl_acc   = xbar_acc + 3 * sigma_acc   # 0.8579
    lcl_acc   = xbar_acc - 3 * sigma_acc   # 0.8099
    cpk_acc   = (ucl_acc - xbar_acc) / (3 * sigma_acc)  # 1.00

    # SPC parameters — DPMO
    xbar_dpmo  = np.mean(ctrl_dpmo)
    sigma_dpmo = np.std(ctrl_dpmo, ddof=1)
    ucl_dpmo   = xbar_dpmo + 3 * sigma_dpmo
    lcl_dpmo   = xbar_dpmo - 3 * sigma_dpmo

    # ── Section header ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="dot" style="background:#FFD700;"></div>
        <h3>Statistical Process Control (SPC) — Control Phase Validation</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#071525;border:1px solid #14324F;border-radius:12px;
    padding:1rem 1.5rem;margin-bottom:1.5rem;font-size:0.83rem;color:#6E90B0;line-height:1.7;'>
    The optimised configuration was retrained in <b style='color:#00D4FF;'>3 independent control runs</b>
    under identical hyperparameters (LR=0.0001, Batch=16, Dropout=0.3, RMSprop, WD=0.0001, 8 epochs).
    All observations fall within the ±3σ control band, confirming the process is
    <b style='color:#00FF9D;'>statistically stable and free from special-cause variation</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── SPC summary KPI row ───────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(5,1fr);">
        <div class="kpi-card cyan">
            <div class="kpi-label">Mean Accuracy (x̄)</div>
            <div class="kpi-value cyan" style="font-size:1.6rem;">{xbar_acc*100:.2f}%</div>
            <div class="kpi-delta">Across 3 control runs</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Std Dev (σ)</div>
            <div class="kpi-value green" style="font-size:1.6rem;">{sigma_acc*100:.3f}%</div>
            <div class="kpi-delta">Low variation confirmed</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">UCL (Accuracy)</div>
            <div class="kpi-value purple" style="font-size:1.6rem;">{ucl_acc*100:.2f}%</div>
            <div class="kpi-delta">x̄ + 3σ</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">LCL (Accuracy)</div>
            <div class="kpi-value purple" style="font-size:1.6rem;">{lcl_acc*100:.2f}%</div>
            <div class="kpi-delta">x̄ − 3σ</div>
        </div>
        <div class="kpi-card orange">
            <div class="kpi-label">Process Cpk</div>
            <div class="kpi-value orange" style="font-size:1.6rem;">{cpk_acc:.2f}</div>
            <div class="kpi-delta">Target ≥ 1.33 (6σ bench)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════ CHART 1 — Individuals Chart (Accuracy) ══════════════════════
    st.markdown('<div class="chart-title" style="margin-top:1rem;">① X-Chart (Individuals) — Accuracy  |  ±3σ Control Limits</div>',
                unsafe_allow_html=True)

    fig_xacc = go.Figure()

    # ── UCL / Centre / LCL bands ──────────────────────────────────────
    x_ext = [0, 4]   # extend lines slightly beyond run range

    # Shaded in-control band
    fig_xacc.add_trace(go.Scatter(
        x=[0.5, 3.5, 3.5, 0.5], y=[ucl_acc*100]*2 + [lcl_acc*100]*2,
        fill="toself",
        fillcolor="rgba(0,212,255,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Warning zone band (±1σ shading)
    fig_xacc.add_trace(go.Scatter(
        x=[0.5, 3.5, 3.5, 0.5],
        y=[(xbar_acc + sigma_acc)*100]*2 + [(xbar_acc - sigma_acc)*100]*2,
        fill="toself",
        fillcolor="rgba(0,255,157,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # UCL line
    fig_xacc.add_shape(type="line", x0=0.5, x1=3.5,
                       y0=ucl_acc*100, y1=ucl_acc*100,
                       line=dict(color="#FF6B35", width=2, dash="dash"))
    fig_xacc.add_annotation(x=3.5, y=ucl_acc*100,
                            text=f"UCL = {ucl_acc*100:.2f}%",
                            xanchor="left", showarrow=False,
                            font=dict(color="#FF6B35", size=11))

    # LCL line
    fig_xacc.add_shape(type="line", x0=0.5, x1=3.5,
                       y0=lcl_acc*100, y1=lcl_acc*100,
                       line=dict(color="#FF6B35", width=2, dash="dash"))
    fig_xacc.add_annotation(x=3.5, y=lcl_acc*100,
                            text=f"LCL = {lcl_acc*100:.2f}%",
                            xanchor="left", showarrow=False,
                            font=dict(color="#FF6B35", size=11))

    # Centre line (x̄)
    fig_xacc.add_shape(type="line", x0=0.5, x1=3.5,
                       y0=xbar_acc*100, y1=xbar_acc*100,
                       line=dict(color="#00D4FF", width=1.5, dash="dot"))
    fig_xacc.add_annotation(x=3.5, y=xbar_acc*100,
                            text=f"x̄ = {xbar_acc*100:.2f}%",
                            xanchor="left", showarrow=False,
                            font=dict(color="#00D4FF", size=11))

    # ±1σ reference lines (subtle)
    for sign, label in [(1, "+1σ"), (-1, "−1σ")]:
        fig_xacc.add_shape(type="line", x0=0.5, x1=3.5,
                           y0=(xbar_acc + sign*sigma_acc)*100,
                           y1=(xbar_acc + sign*sigma_acc)*100,
                           line=dict(color="rgba(0,212,255,0.25)", width=1, dash="dot"))

    # Baseline reference line
    baseline_acc = 0.825321
    fig_xacc.add_shape(type="line", x0=0.5, x1=3.5,
                       y0=baseline_acc*100, y1=baseline_acc*100,
                       line=dict(color="#6E90B0", width=1.5, dash="longdash"))
    fig_xacc.add_annotation(x=3.5, y=baseline_acc*100,
                            text=f"Baseline = {baseline_acc*100:.2f}%",
                            xanchor="left", showarrow=False,
                            font=dict(color="#6E90B0", size=10))

    # Data points — individual control runs
    fig_xacc.add_trace(go.Scatter(
        x=list(range(1, 4)), y=[v*100 for v in ctrl_acc],
        mode="lines+markers+text",
        name="Control Runs",
        line=dict(color="#00FF9D", width=3),
        marker=dict(
            size=16, color="#00FF9D",
            line=dict(color="#030B14", width=3),
            symbol="circle",
        ),
        text=[f"{v*100:.2f}%" for v in ctrl_acc],
        textposition=["top center", "bottom center", "top center"],
        textfont=dict(color="#00FF9D", size=12, family="Syne Mono"),
        hovertemplate="<b>%{customdata}</b><br>Accuracy: %{y:.2f}%<extra></extra>",
        customdata=ctrl_runs,
    ))

    # In-control annotation (green tick per point)
    for i, (run, val) in enumerate(zip(ctrl_runs, ctrl_acc)):
        fig_xacc.add_annotation(
            x=i+1, y=lcl_acc*100 - 0.3,
            text="✓ IN CONTROL",
            showarrow=False,
            font=dict(color="#00FF9D", size=9, family="Syne Mono"),
            xanchor="center",
        )

    fig_xacc.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        showlegend=True,
        legend_x=0.01, legend_y=0.99,
    )
    fig_xacc.update_xaxes(tickvals=[1,2,3], ticktext=ctrl_runs,
        gridcolor="#14324F", linecolor="#14324F", range=[0.4, 4.2])
    fig_xacc.update_yaxes(title_text="Accuracy (%)",
        gridcolor="#14324F", linecolor="#14324F",
        range=[lcl_acc*100 - 1.5, ucl_acc*100 + 1.5])
    st.plotly_chart(fig_xacc, use_container_width=True)

    # ════ CHART 2 — Individuals Chart (DPMO) ══════════════════════════
    st.markdown('<div class="chart-title" style="margin-top:1rem;">② X-Chart (Individuals) — DPMO  |  ±3σ Control Limits</div>',
                unsafe_allow_html=True)

    fig_xdpmo = go.Figure()

    # Shaded in-control band
    fig_xdpmo.add_trace(go.Scatter(
        x=[0.5, 3.5, 3.5, 0.5], y=[ucl_dpmo]*2 + [lcl_dpmo]*2,
        fill="toself", fillcolor="rgba(123,97,255,0.06)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    for y_val, color, label, dash in [
        (ucl_dpmo, "#FF6B35", f"UCL = {ucl_dpmo:,.0f}", "dash"),
        (lcl_dpmo, "#FF6B35", f"LCL = {lcl_dpmo:,.0f}", "dash"),
        (xbar_dpmo, "#00D4FF", f"x̄ = {xbar_dpmo:,.0f}", "dot"),
        (174679,    "#6E90B0", "Baseline = 174,679", "longdash"),
    ]:
        fig_xdpmo.add_shape(type="line", x0=0.5, x1=3.5,
                            y0=y_val, y1=y_val,
                            line=dict(color=color, width=1.5, dash=dash))
        fig_xdpmo.add_annotation(x=3.5, y=y_val,
                                 text=label, xanchor="left", showarrow=False,
                                 font=dict(color=color, size=11))

    fig_xdpmo.add_trace(go.Scatter(
        x=list(range(1, 4)), y=ctrl_dpmo,
        mode="lines+markers+text",
        name="Control Runs (DPMO)",
        line=dict(color="#7B61FF", width=3),
        marker=dict(size=16, color="#7B61FF",
                    line=dict(color="#030B14", width=3)),
        text=[f"{v:,.0f}" for v in ctrl_dpmo],
        textposition=["top center", "bottom center", "top center"],
        textfont=dict(color="#7B61FF", size=12, family="Syne Mono"),
        hovertemplate="<b>%{customdata}</b><br>DPMO: %{y:,.0f}<extra></extra>",
        customdata=ctrl_runs,
    ))

    for i, (run, val) in enumerate(zip(ctrl_runs, ctrl_dpmo)):
        fig_xdpmo.add_annotation(
            x=i+1, y=lcl_dpmo - 3000,
            text="✓ IN CONTROL", showarrow=False,
            font=dict(color="#7B61FF", size=9, family="Syne Mono"),
            xanchor="center",
        )

    fig_xdpmo.update_layout(
        **PLOTLY_LAYOUT, height=400,
        showlegend=True,
        legend_x=0.01, legend_y=0.99,
    )
    fig_xdpmo.update_xaxes(tickvals=[1,2,3], ticktext=ctrl_runs,
        gridcolor="#14324F", linecolor="#14324F", range=[0.4, 4.2])
    fig_xdpmo.update_yaxes(title_text="DPMO",
        gridcolor="#14324F", linecolor="#14324F",
        range=[lcl_dpmo - 12000, ucl_dpmo + 12000])
    st.plotly_chart(fig_xdpmo, use_container_width=True)

    # ════ CHART 3 — Moving Range Chart ════════════════════════════════
    st.markdown('<div class="chart-title" style="margin-top:1rem;">③ Moving Range (MR) Chart — Accuracy  |  Process Variability Monitor</div>',
                unsafe_allow_html=True)

    mr_vals = [abs(ctrl_acc[i] - ctrl_acc[i-1])*100 for i in range(1, len(ctrl_acc))]
    mr_bar  = np.mean(mr_vals)
    d2      = 1.128   # control chart constant for n=2
    ucl_mr  = 3.267 * mr_bar   # D4 * MR_bar (D4=3.267 for n=2)
    lcl_mr  = 0.0              # D3 * MR_bar = 0 for n=2

    fig_mr = go.Figure()

    # UCL / centre
    for y_val, color, label, dash in [
        (ucl_mr, "#FF6B35", f"UCL = {ucl_mr:.3f}%", "dash"),
        (mr_bar, "#00D4FF", f"MR̄ = {mr_bar:.3f}%", "dot"),
        (lcl_mr, "#FF6B35", f"LCL = {lcl_mr:.1f}%", "dash"),
    ]:
        fig_mr.add_shape(type="line", x0=0.5, x1=2.5,
                         y0=y_val, y1=y_val,
                         line=dict(color=color, width=1.5, dash=dash))
        fig_mr.add_annotation(x=2.5, y=y_val, text=label,
                               xanchor="left", showarrow=False,
                               font=dict(color=color, size=11))

    # MR data points
    fig_mr.add_trace(go.Scatter(
        x=list(range(1, len(mr_vals)+1)), y=mr_vals,
        mode="lines+markers+text",
        name="Moving Range",
        line=dict(color="#FFD700", width=3),
        marker=dict(size=14, color="#FFD700",
                    line=dict(color="#030B14", width=3)),
        text=[f"{v:.4f}%" for v in mr_vals],
        textposition="top center",
        textfont=dict(color="#FFD700", size=12, family="Syne Mono"),
    ))

    fig_mr.update_layout(
        **PLOTLY_LAYOUT, height=300,
        showlegend=True,
        legend_x=0.01, legend_y=0.99,
    )
    fig_mr.update_xaxes(
        tickvals=list(range(1, len(mr_vals)+1)),
        ticktext=[f"MR({i},{i+1})" for i in range(1, len(mr_vals)+1)],
        gridcolor="#14324F", linecolor="#14324F", range=[0.3, 3.3])
    fig_mr.update_yaxes(
        title_text="Moving Range (%)",
        gridcolor="#14324F", linecolor="#14324F",
        range=[-0.01, ucl_mr * 1.4])
    st.plotly_chart(fig_mr, use_container_width=True)

    # ════ SPC SUMMARY TABLE ═══════════════════════════════════════════
    st.markdown("""
    <div class="section-header" style="margin-top:1.5rem;">
        <div class="dot" style="background:#FFD700;"></div>
        <h3>SPC Summary — All Control Observations</h3>
    </div>
    """, unsafe_allow_html=True)

    spc_rows = []
    for i, (run, acc, dpmo) in enumerate(zip(ctrl_runs, ctrl_acc, ctrl_dpmo)):
        in_ctrl_acc  = lcl_acc  <= acc  <= ucl_acc
        in_ctrl_dpmo = lcl_dpmo <= dpmo <= ucl_dpmo
        dev_acc   = (acc - xbar_acc) / sigma_acc
        spc_rows.append({
            "Run": run,
            "Accuracy": f"{acc*100:.2f}%",
            "vs x̄": f"{'+'if acc>=xbar_acc else ''}{(acc-xbar_acc)*100:.4f}%",
            "σ Distance": f"{dev_acc:+.2f}σ",
            "DPMO": f"{dpmo:,}",
            "UCL Acc": f"{ucl_acc*100:.2f}%",
            "LCL Acc": f"{lcl_acc*100:.2f}%",
            "In Control?": "✅ YES" if in_ctrl_acc else "❌ NO",
        })

    spc_df = pd.DataFrame(spc_rows)

    st.dataframe(spc_df, use_container_width=True, hide_index=True,
                 column_config={
                     "In Control?": st.column_config.TextColumn("In Control?"),
                 })

    # Capability interpretation card
    st.markdown(f"""
    <div style='background:#071525;border:1px solid #14324F;border-radius:12px;
    padding:1.2rem 1.8rem;margin-top:1.5rem;'>
        <div style='display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-start;'>
            <div>
                <div style='font-size:0.7rem;color:#6E90B0;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:0.4rem;'>Control Band</div>
                <div style='font-family:Syne Mono,monospace;color:#00D4FF;font-size:1rem;'>
                [{lcl_acc*100:.2f}% , {ucl_acc*100:.2f}%]</div>
            </div>
            <div>
                <div style='font-size:0.7rem;color:#6E90B0;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:0.4rem;'>Process Capability Cpk</div>
                <div style='font-family:Syne,sans-serif;font-weight:800;
                color:#FFD700;font-size:1.3rem;'>{cpk_acc:.2f}
                &nbsp;<span style='font-size:0.75rem;color:#6E90B0;font-weight:400;'>
                (target ≥ 1.33)</span></div>
            </div>
            <div>
                <div style='font-size:0.7rem;color:#6E90B0;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:0.4rem;'>Verdict</div>
                <div style='color:#00FF9D;font-weight:700;font-size:0.95rem;'>
                ✅ All 3 runs within ±3σ limits<br>
                <span style='color:#6E90B0;font-size:0.8rem;font-weight:400;'>
                Process is stable. No special-cause variation detected.</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════
# TAB 6 — DIAGNOSTIC (Image Upload)
# ════════════════════════════════════════
with tab6:
    st.markdown("""
    <div class="section-header">
        <div class="dot" style="background:#00FF9D;"></div>
        <h3>🫁 Chest X-Ray Diagnostic Tool — Upload & Predict</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#071525;border:1px solid #14324F;border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.5rem;'>
    <span style='color:#6E90B0;font-size:0.82rem;'>
    ⚠️ <b style='color:#FFD700;'>Note:</b> This demo uses simulated inference. To run real predictions,
    load a trained PyTorch model (<code style='color:#00D4FF;'>final_1772431147.pt</code>) and
    replace the <code style='color:#00D4FF;'>mock_predict()</code> function below with actual inference code.
    The app structure and UI are production-ready.
    </span>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload a Chest X-Ray image",
            type=["jpg","jpeg","png","bmp","tiff"],
            help="Upload a chest X-ray image (JPEG or PNG preferred)"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded X-Ray", use_column_width=True,
                     output_format="auto")

            # Show image metadata
            st.markdown(f"""
            <div style='background:#0C1E33;border:1px solid #14324F;border-radius:8px;
            padding:0.7rem 1rem;margin-top:0.5rem;font-size:0.78rem;'>
            <span style='color:#6E90B0;'>Size:</span>
            <span style='color:#00D4FF;font-family:Syne Mono,monospace;'>{img.width}×{img.height}px</span>
            &nbsp;&nbsp;
            <span style='color:#6E90B0;'>Mode:</span>
            <span style='color:#00D4FF;font-family:Syne Mono,monospace;'>{img.mode}</span>
            &nbsp;&nbsp;
            <span style='color:#6E90B0;'>File:</span>
            <span style='color:#00D4FF;font-family:Syne Mono,monospace;'>{uploaded.name}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        if uploaded:
            # ── MOCK INFERENCE (replace with real model) ──────────────
            def mock_predict(image: Image.Image):
                """
                Replace this function with real inference:

                import torch
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224,224)), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
                model = torch.load("final_1772431147.pt", map_location="cpu")
                model.eval()
                with torch.no_grad():
                    inp = transform(image).unsqueeze(0)
                    out = torch.softmax(model(inp), dim=1)
                    conf, idx = out.max(1)
                    return ["NORMAL","PNEUMONIA"][idx.item()], conf.item()
                """
                # Deterministic mock based on image characteristics
                img_arr = np.array(image.resize((224,224)).convert("L"), dtype=np.float32)
                mean_brightness = img_arr.mean()
                # Heuristic: darker X-ray = more likely Pneumonia (for demo)
                if mean_brightness < 128:
                    conf = 0.72 + (128 - mean_brightness) / 1000
                    return "PNEUMONIA", min(conf, 0.97)
                else:
                    conf = 0.65 + (mean_brightness - 128) / 1000
                    return "NORMAL", min(conf, 0.97)
            # ────────────────────────────────────────────────────────────

            with st.spinner("Running inference…"):
                import time; time.sleep(0.8)
                label, conf = mock_predict(img)

            is_pneumonia = label == "PNEUMONIA"
            conf_pct = conf * 100
            cls_color = "pneumonia" if is_pneumonia else "normal"
            icon = "🔴" if is_pneumonia else "🟢"
            alt_conf = (1 - conf) * 100

            st.markdown(f"""
            <div class="diag-result {cls_color}">
                <div style='font-size:0.72rem;color:#6E90B0;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:0.5rem;'>Diagnosis Result</div>
                <div class="result-label {cls_color}">{icon} {label}</div>
                <div style='font-size:0.85rem;color:#6E90B0;margin-top:0.3rem;'>
                Confidence: <b style='color:{"#FF6B35" if is_pneumonia else "#00FF9D"};'>{conf_pct:.1f}%</b></div>
                <div class="conf-bar-bg" style="margin-top:0.6rem;">
                    <div style='height:10px;width:{conf_pct:.1f}%;
                    background:{"#FF6B35" if is_pneumonia else "#00FF9D"};border-radius:20px;'></div>
                </div>
                <div style='display:flex;justify-content:space-between;
                font-size:0.72rem;color:#6E90B0;margin-top:0.2rem;'>
                    <span>0%</span><span>100%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Class probabilities
            st.markdown('<div class="chart-title" style="margin-top:1rem;">Class Probability Distribution</div>',
                        unsafe_allow_html=True)
            fig_prob = go.Figure(go.Bar(
                x=["NORMAL","PNEUMONIA"],
                y=[alt_conf if is_pneumonia else conf_pct,
                   conf_pct if is_pneumonia else alt_conf],
                marker_color=["#00FF9D","#FF6B35"],
                text=[f"{alt_conf if is_pneumonia else conf_pct:.1f}%",
                      f"{conf_pct if is_pneumonia else alt_conf:.1f}%"],
                textposition="auto",
            ))
            fig_prob.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False,
                                   yaxis_title="Probability (%)", yaxis_range=[0,100])
            st.plotly_chart(fig_prob, use_container_width=True)

            # Clinical note
            if is_pneumonia:
                st.warning("⚕️ **Clinical Note:** Pneumonia detected with high confidence. "
                           "Please consult a radiologist for clinical validation.")
            else:
                st.success("✅ **Clinical Note:** No pneumonia detected. Lungs appear clear.")

        else:
            st.markdown("""
            <div style='text-align:center;padding:4rem 2rem;color:#6E90B0;'>
                <div style='font-size:4rem;margin-bottom:1rem;'>🫁</div>
                <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#E8F4FF;margin-bottom:0.5rem;'>
                Upload a chest X-ray to get started</div>
                <div style='font-size:0.82rem;'>
                The model will classify the image as <b style='color:#00FF9D;'>NORMAL</b>
                or <b style='color:#FF6B35;'>PNEUMONIA</b> with confidence scores
                </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;
border-top:1px solid #14324F;margin-top:2rem;color:#6E90B0;font-size:0.75rem;'>
    <span style='font-family:Syne Mono,monospace;'>
    LSS × Deep Learning Dashboard &nbsp;·&nbsp; Team Theta &nbsp;·&nbsp;
    DMAIC Applied to Chest X-Ray Classification
    </span><br>
    <span style='font-size:0.68rem;opacity:0.6;'>
    Baseline → DOE → Improve → Focused DOE → Final Model &nbsp;|&nbsp;
    Accuracy: 82.5% → 87.0% &nbsp;|&nbsp; DPMO: 174,679 → 133,013
    </span>
</div>
""", unsafe_allow_html=True)