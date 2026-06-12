"""
=============================================================================
 GAMSAT SCORE PREDICTOR v2 — Combined Admissions Analytics Dashboard
=============================================================================
 Architecture (single-file, modular):
   1. DATA LAYER      — ground-truth lookup matrices (from consensus config)
   2. MATH ENGINE     — vectorized NumPy interpolation + SciPy curve fitting
                        + Combo Score (GAMSAT/100 + GPA/7) modeling
   3. UI LAYER        — Streamlit sidebar controls + main results dashboard

 v2 upgrades:
   - GPA slider (4.00–7.00, 0.05 step) on the 7-point Australian scale
   - Casper Performance quartile selector
   - "Combined Admissions Competitiveness" summary with Combo Scores
   - Dynamic conditional alerts for Casper-weighted institutions

 Performance: all interpolation is vectorized (np.interp); the global
 percentile curve fit (scipy.optimize) is cached and runs once per session.

 DISCLAIMER: This tool models *consensus community estimates*. It is not
 affiliated with ACER, Acuity Insights (Casper), or GEMSAS, and produces
 indicative estimates only.
=============================================================================
"""

# --- Required imports (declared explicitly at top of file) -----------------
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import plotly.graph_objects as go


# ============================================================================
# 1. DATA LAYER — ground-truth reference matrices
# ============================================================================

# Section 1 (Reasoning in Humanities): raw percentile -> scaled estimate
S1_RAW = np.array([10, 20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float)
S1_SCALED = np.array([40, 44, 48, 52, 56, 60, 62, 65, 68, 72, 75, 79, 84, 90], dtype=float)

# Section 3 (Reasoning in Sciences): raw percentile -> scaled estimate (S-curve)
S3_RAW = np.array([10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float)
S3_SCALED = np.array([38, 42, 46, 50, 55, 58, 61, 64, 68, 71, 75, 79, 83, 88, 95], dtype=float)

# Section 2 (Written Communication): rubric band -> consensus scaled median
S2_BANDS = {
    "Band 1: Basic / Formulaic": {
        "median": 49.0,
        "desc": "Underdeveloped arguments, simplistic vocabulary, or heavy reliance on rigid essay templates.",
    },
    "Band 2: Competent / Structured": {
        "median": 58.5,
        "desc": "Clear paragraph structure, addresses themes directly, strong mechanical control, predictable arguments.",
    },
    "Band 3: Advanced / Analytical": {
        "median": 69.5,
        "desc": "Unique thesis statement, notable conceptual depth, excellent linguistic nuance, sophisticated sentence structures.",
    },
    "Band 4: Exceptional / Philosophical": {
        "median": 80.0,
        "desc": "Highly mature authorial voice, flawless expression, fluidly synthesizes complex human conditions or abstract ideologies.",
    },
}

# Overall score -> global percentile (benchmark distribution anchor nodes)
PCTL_SCORES = np.array([50, 54, 58, 61, 63, 66, 68, 70, 73, 76], dtype=float)
PCTL_VALUES = np.array([15, 30, 50, 65, 75, 83, 90, 95, 98, 99.5], dtype=float)

# Casper Performance quartile options
CASPER_QUARTILES = [
    "1st Quartile (Bottom 25%)",
    "2nd Quartile",
    "3rd Quartile",
    "4th Quartile (Top 25%)",
]

# Australian medical school admissions rule matrix
# Fields: name, method, gamsat_cutoff, hurdle_text, min_section, min_overall,
#         uses_combo (GAMSAT/100 + GPA/7 ranking), gpa_hurdle_only,
#         casper_weighted
UNI_RULES = [
    {
        "name": "University of Sydney (USyd)",
        "method": "Section-by-Section Rank",
        "cutoff": 68.0,
        "hurdle": "Min 50 in each section. GPA is a hurdle only (≥ 5.0) — it does not contribute to the ranking score.",
        "min_section": 50, "min_overall": None,
        "uses_combo": False, "gpa_hurdle": 5.0, "casper_weighted": False,
    },
    {
        "name": "University of Melbourne (UoM)",
        "method": "Unweighted Average",
        "cutoff": 67.0,
        "hurdle": "Min 50 in each individual section.",
        "min_section": 50, "min_overall": None,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": False,
    },
    {
        "name": "University of Queensland (UQ)",
        "method": "Unweighted Average",
        "cutoff": 67.0,
        "hurdle": "Min 50 in each individual section.",
        "min_section": 50, "min_overall": None,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": False,
    },
    {
        "name": "Notre Dame Sydney (UNDS)",
        "method": "Unweighted Average",
        "cutoff": 63.0,
        "hurdle": "Min 50 in each section & min overall score of 52.",
        "min_section": 50, "min_overall": 52,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": True,
    },
    {
        "name": "Notre Dame Fremantle (UNDF)",
        "method": "Unweighted Average",
        "cutoff": 63.0,
        "hurdle": "Min 50 in each section & min overall score of 52.",
        "min_section": 50, "min_overall": 52,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": True,
    },
    {
        "name": "Australian National University (ANU)",
        "method": "Weighted Average",
        "cutoff": 66.0,
        "hurdle": "Min 50 in each section & min overall score of 50.",
        "min_section": 50, "min_overall": 50,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": False,
    },
    {
        "name": "Deakin University",
        "method": "Weighted Average",
        "cutoff": 66.0,
        "hurdle": "Min 50 in each section & min overall score of 50.",
        "min_section": 50, "min_overall": 50,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": False,
    },
    {
        "name": "Griffith University",
        "method": "Weighted Average",
        "cutoff": 64.0,
        "hurdle": "Min 50 in each individual section.",
        "min_section": 50, "min_overall": None,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": False,
    },
    {
        "name": "Macquarie University",
        "method": "Weighted Average",
        "cutoff": 65.0,
        "hurdle": "Min 50 in each section & min overall score of 50.",
        "min_section": 50, "min_overall": 50,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": False,
    },
    {
        "name": "University of Western Australia (UWA)",
        "method": "Weighted Average",
        "cutoff": 65.0,
        "hurdle": "Min 50 in each section & min overall score of 55.",
        "min_section": 50, "min_overall": 55,
        "uses_combo": True, "gpa_hurdle": None, "casper_weighted": False,
    },
    {
        "name": "University of Wollongong (UoW)",
        "method": "Weighted Average",
        "cutoff": 65.0,
        "hurdle": "GAMSAT used strictly as a qualifying hurdle baseline entry.",
        "min_section": 50, "min_overall": None,
        "uses_combo": False, "gpa_hurdle": None, "casper_weighted": True,
    },
]


# ============================================================================
# 2. MATH ENGINE — interpolation, distribution fit, and Combo Score model
# ============================================================================

def interpolate_scaled(raw_pct: float, raw_nodes: np.ndarray, scaled_nodes: np.ndarray) -> float:
    """Fluid non-linear interpolation of a raw percentile onto the scaled
    score curve (vectorized C-level np.interp — instant on slider movement)."""
    return float(np.interp(raw_pct, raw_nodes, scaled_nodes))


def _normal_cdf_pct(x, mu, sigma):
    """Normal CDF expressed in percent — model function for curve fitting."""
    return norm.cdf(x, loc=mu, scale=sigma) * 100.0


@st.cache_data(show_spinner=False)
def fit_global_distribution():
    """Fit a Normal CDF to the benchmark percentile anchor nodes via
    scipy.optimize.curve_fit. Cached — the optimization runs once per session."""
    (mu, sigma), _ = curve_fit(_normal_cdf_pct, PCTL_SCORES, PCTL_VALUES, p0=[58.0, 8.0])
    return float(mu), float(sigma)


def overall_scores(s1: float, s2: float, s3: float) -> tuple[float, float]:
    """Return (unweighted, weighted) overall GAMSAT scores.
    Unweighted: (S1 + S2 + S3) / 3.   Weighted: (S1 + S2 + 2·S3) / 4."""
    return (s1 + s2 + s3) / 3.0, (s1 + s2 + 2.0 * s3) / 4.0


def percentile_lookup(score: float) -> float:
    """Interpolate the global percentile from the anchor-node matrix."""
    return float(np.interp(score, PCTL_SCORES, PCTL_VALUES))


def combo_score(gamsat: float, gpa: float) -> float:
    """Standard GEMSAS-style Combo Score: (GAMSAT / 100) + (GPA / 7).
    Theoretical maximum 2.00; competitive offers typically cluster ≥ ~1.60."""
    return (gamsat / 100.0) + (gpa / 7.0)


def evaluate_university(rule: dict, s1, s2, s3, unweighted, weighted, gpa, casper_idx):
    """Evaluate one university against the full candidate profile.
    Returns dict with relevant GAMSAT score, combo score (or None),
    meets-all-criteria flag, and a Casper relevance flag."""
    sections_ok = min(s1, s2, s3) >= rule["min_section"]

    gamsat_score = weighted if rule["method"] == "Weighted Average" else unweighted

    overall_ok = (rule["min_overall"] is None) or (gamsat_score >= rule["min_overall"])
    cutoff_ok = gamsat_score >= rule["cutoff"]
    gpa_ok = (rule["gpa_hurdle"] is None) or (gpa >= rule["gpa_hurdle"])

    cs = combo_score(gamsat_score, gpa) if rule["uses_combo"] else None

    return {
        "gamsat_score": gamsat_score,
        "combo": cs,
        "meets": sections_ok and overall_ok and cutoff_ok and gpa_ok,
        "casper_weighted": rule["casper_weighted"],
    }


# ============================================================================
# 3. UI LAYER — Streamlit dashboard
# ============================================================================

st.set_page_config(
    page_title="GAMSAT Score Predictor",
    page_icon="📊",
    layout="wide",
)

# ---- Sidebar: all raw data controls ----------------------------------------
with st.sidebar:
    st.header("Input Controls")
    st.caption("Adjust your estimated performance profile below.")

    st.subheader("Section 1 — Humanities")
    s1_raw = st.slider(
        "Raw percentile (practice materials)",
        min_value=10, max_value=100, value=65, step=1, key="s1",
    )

    st.subheader("Section 2 — Written Communication")
    s2_band = st.selectbox(
        "Essay rubric band",
        options=list(S2_BANDS.keys()),
        index=1,
    )
    st.caption(S2_BANDS[s2_band]["desc"])

    st.subheader("Section 3 — Sciences")
    s3_raw = st.slider(
        "Raw percentile (practice materials)",
        min_value=10, max_value=100, value=65, step=1, key="s3",
    )

    st.divider()

    st.subheader("Academic Record")
    gpa = st.slider(
        "GPA (7-point scale)",
        min_value=4.00, max_value=7.00, value=6.00, step=0.05,
        format="%.2f",
        help="GEMSAS-weighted GPA on the standard Australian 7-point scale.",
    )

    st.subheader("Casper Performance")
    casper_choice = st.selectbox(
        "Estimated quartile placement",
        options=CASPER_QUARTILES,
        index=2,
        help="Casper results are reported as quartile placements. Relevant to UoW and Notre Dame.",
    )
    casper_idx = CASPER_QUARTILES.index(casper_choice)  # 0..3

    st.divider()
    st.caption(
        "Estimates are modeled from historical consensus data and are "
        "indicative only. Not affiliated with ACER, Acuity Insights, or GEMSAS."
    )

# ---- Math pipeline (all vectorized, instant) --------------------------------
s1_scaled = interpolate_scaled(s1_raw, S1_RAW, S1_SCALED)
s2_scaled = S2_BANDS[s2_band]["median"]
s3_scaled = interpolate_scaled(s3_raw, S3_RAW, S3_SCALED)

unweighted, weighted = overall_scores(s1_scaled, s2_scaled, s3_scaled)
pctl = percentile_lookup(unweighted)
mu, sigma = fit_global_distribution()

combo_unweighted = combo_score(unweighted, gpa)
combo_weighted = combo_score(weighted, gpa)

# ---- Main pane: high-impact outputs -----------------------------------------
st.title("GAMSAT Score Predictor")
st.caption("Psychometric exam analytics · combined admissions competitiveness modeling")

st.divider()

# Section-level scaled estimates
st.subheader("Scaled Section Estimates")
c1, c2, c3 = st.columns(3)
c1.metric("Section 1 · Humanities", f"{s1_scaled:.1f}")
c2.metric("Section 2 · Written Comm.", f"{s2_scaled:.1f}")
c3.metric("Section 3 · Sciences", f"{s3_scaled:.1f}")

st.divider()

# Overall GAMSAT models — side-by-side on desktop, stacked on mobile
st.subheader("Overall Score Models")
col_u, col_w = st.columns(2)
with col_u:
    st.metric(
        "Unweighted Average",
        f"{unweighted:.1f}",
        help="(S1 + S2 + S3) ÷ 3 — used by UoM, UQ, Notre Dame.",
    )
with col_w:
    st.metric(
        "Weighted Average (S3 ×2)",
        f"{weighted:.1f}",
        delta=f"{weighted - unweighted:+.1f} vs unweighted",
        help="(S1 + S2 + 2·S3) ÷ 4 — used by ANU, Deakin, Griffith, Macquarie, UWA, UoW.",
    )

st.info(
    f"**Global standing:** an unweighted score of **{unweighted:.1f}** maps to "
    f"approximately the **{pctl:.0f}th percentile** of the benchmark cohort "
    f"distribution (fitted model: μ = {mu:.1f}, σ = {sigma:.1f})."
)

st.divider()

# ---- Combined Admissions Competitiveness ------------------------------------
st.subheader("Combined Admissions Competitiveness")
st.caption(
    "Combo Score model: (GAMSAT ÷ 100) + (GPA ÷ 7). Theoretical maximum 2.00. "
    "Applies to combo-ranked institutions; USyd treats GPA strictly as a "
    "hurdle (≥ 5.0) with no contribution to ranking."
)

cc1, cc2, cc3 = st.columns(3)
cc1.metric(
    "Combo (Unweighted GAMSAT)",
    f"{combo_unweighted:.3f}",
    help="(Unweighted GAMSAT ÷ 100) + (GPA ÷ 7).",
)
cc2.metric(
    "Combo (Weighted GAMSAT)",
    f"{combo_weighted:.3f}",
    delta=f"{combo_weighted - combo_unweighted:+.3f}",
    help="(Weighted GAMSAT ÷ 100) + (GPA ÷ 7).",
)
cc3.metric(
    "GPA Contribution",
    f"{gpa / 7.0:.3f}",
    help=f"GPA {gpa:.2f} ÷ 7 — the academic-record share of the Combo Score.",
)

# Dynamic conditional alerts -------------------------------------------------
# GPA hurdle check for USyd
if gpa < 5.0:
    st.warning(
        f"**USyd GPA hurdle not met:** your GPA of {gpa:.2f} is below the 5.0 "
        "hurdle. At USyd, GPA does not affect ranking, but the hurdle must be "
        "cleared to be considered."
    )
else:
    st.info(
        f"**USyd note:** GPA {gpa:.2f} clears the 5.0 hurdle. Beyond the hurdle, "
        "GPA adds nothing at USyd — ranking is driven by section-by-section "
        "GAMSAT performance alone."
    )

# Casper-driven alert for Casper-weighted institutions (UoW, Notre Dame)
if casper_idx >= 2:  # 3rd or 4th quartile
    st.success(
        f"**Strong Casper position ({casper_choice}):** at Casper-weighted "
        "institutions — UoW and Notre Dame (UNDS/UNDF) — an upper-quartile "
        "placement is a significant competitive lever, since these programs "
        "weight Casper heavily once GAMSAT/GPA hurdles are cleared. Your "
        "profile is well positioned there."
    )
elif casper_idx == 1:
    st.info(
        f"**Casper position ({casper_choice}):** a mid-range placement keeps "
        "UoW and Notre Dame in play but is unlikely to be a differentiator. "
        "Combo-ranked institutions may suit your profile better."
    )
else:
    st.warning(
        f"**Casper position ({casper_choice}):** a bottom-quartile placement "
        "is a material disadvantage at UoW and Notre Dame, where Casper "
        "carries heavy selection weight. Combo-ranked institutions driven by "
        "GAMSAT + GPA will likely serve your profile better."
    )

st.divider()

# ---- Plotly bell curve -------------------------------------------------------
st.subheader("Cohort Distribution Position")

x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
y = norm.pdf(x, mu, sigma)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y, mode="lines", name="Cohort distribution",
    line=dict(color="#2C6E8F", width=2.5),
    fill="tozeroy", fillcolor="rgba(44,110,143,0.08)",
    hovertemplate="Score %{x:.1f}<extra></extra>",
))
for k, shade in [(1, "rgba(44,110,143,0.10)"), (2, "rgba(44,110,143,0.05)")]:
    fig.add_vrect(x0=mu - k * sigma, x1=mu + k * sigma,
                  fillcolor=shade, line_width=0, layer="below")
for k in (-2, -1, 0, 1, 2):
    fig.add_vline(
        x=mu + k * sigma, line_dash="dot",
        line_color="rgba(60,60,60,0.25)",
        annotation_text=("μ" if k == 0 else f"{k:+d}σ"),
        annotation_position="top",
        annotation_font=dict(size=11, color="#6B7280"),
    )

user_z = (unweighted - mu) / sigma
fig.add_trace(go.Scatter(
    x=[unweighted], y=[norm.pdf(unweighted, mu, sigma)],
    mode="markers+text", name="Your estimate",
    marker=dict(size=14, color="#1B7F4D", symbol="diamond",
                line=dict(width=2, color="white")),
    text=[f"  {unweighted:.1f} ({user_z:+.2f}σ)"],
    textposition="middle right",
    textfont=dict(size=13, color="#1B7F4D"),
    hovertemplate=f"Your score: {unweighted:.1f}<br>z = {user_z:+.2f}<extra></extra>",
))
fig.update_layout(
    margin=dict(l=10, r=10, t=30, b=10),
    height=380,
    showlegend=False,
    xaxis_title="Overall scaled score",
    yaxis=dict(showticklabels=False, showgrid=False, title=None),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Sans Pro, sans-serif"),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- University outlook matrix -----------------------------------------------
st.subheader("University Outlook Matrix")
st.caption(
    "Rows highlighted green where your modeled profile satisfies the cut-off "
    "and all hurdle requirements (GAMSAT sections, overall floors, GPA "
    "hurdles) under that institution's calculation method."
)

rows, meets_flags = [], []
for rule in UNI_RULES:
    result = evaluate_university(rule, s1_scaled, s2_scaled, s3_scaled,
                                 unweighted, weighted, gpa, casper_idx)
    rows.append({
        "University": rule["name"],
        "Method": rule["method"],
        "Cut-off": f"{rule['cutoff']:.1f}",
        "Your GAMSAT": f"{result['gamsat_score']:.1f}",
        "Combo Score": f"{result['combo']:.3f}" if result["combo"] is not None else "n/a",
        "Casper": "Weighted" if result["casper_weighted"] else "—",
        "Status": "✅ Meets benchmark" if result["meets"] else "— Below benchmark",
        "Hurdle Requirement": rule["hurdle"],
    })
    meets_flags.append(result["meets"])

uni_df = pd.DataFrame(rows)


def _highlight_meets(row):
    """Row styler: soft clinical green where all criteria are satisfied."""
    if meets_flags[row.name]:
        return ["background-color: #E7F4EC; color: #14532D"] * len(row)
    return [""] * len(row)


st.dataframe(
    uni_df.style.apply(_highlight_meets, axis=1),
    use_container_width=True,
    hide_index=True,
    height=430,
)

n_meets = sum(meets_flags)
if n_meets:
    st.success(
        f"Your modeled profile currently satisfies the standard competitive "
        f"benchmark at **{n_meets} of {len(UNI_RULES)}** institutions listed."
    )
else:
    st.info(
        "Your modeled profile does not yet reach the standard competitive "
        "benchmarks listed. Section 3 carries double weight at several "
        "institutions — gains there move the weighted average fastest."
    )

st.caption(
    "Notes: USyd ranks section-by-section (overall averages not used; GPA is "
    "a ≥ 5.0 hurdle only) — its row is shown against the unweighted mean for "
    "indicative comparison. UoW uses GAMSAT strictly as a qualifying hurdle, "
    "with Casper and other factors driving selection. Cut-offs shift annually "
    "with each applicant pool — always verify against current GEMSAS and "
    "university admissions guides."
)

# ----------------------------------------------------------------------------
# To run this app, execute the following in your terminal:
# streamlit run app.py
# ----------------------------------------------------------------------------