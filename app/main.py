"""
RenalGuard AI - Main Streamlit Application
AI-Powered Early Detection & Clinical Decision Support for Chronic Kidney Disease
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ─── PATH SETUP ──────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RenalGuard AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── INLINE CSS (styling only — no layout wrappers) ──────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 100%);
}

/* Hide Streamlit default top padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* App header brand */
.rg-brand {
    text-align: center;
    padding: 0.5rem 0 1rem 0;
}
.rg-brand h1 {
    font-size: 2rem;
    font-weight: 900;
    color: #4F46E5;
    margin: 0;
    letter-spacing: -0.03em;
}
.rg-brand p {
    color: #6B7280;
    font-size: 0.85rem;
    margin: 0;
}

/* Hero */
.rg-hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #1E293B 0%, #4F46E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.05;
    letter-spacing: -0.04em;
    margin-bottom: 0.5rem;
}
.rg-hero-sub {
    font-size: 1.1rem;
    color: #64748B;
    max-width: 550px;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* Metric tiles */
.rg-metric {
    background: white;
    border-radius: 20px;
    padding: 1.5rem 1rem;
    text-align: center;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(99,102,241,0.06);
    transition: transform 0.2s;
}
.rg-metric:hover { transform: translateY(-3px); }
.rg-metric-val {
    font-size: 2.2rem;
    font-weight: 900;
    color: #4F46E5;
    letter-spacing: -0.03em;
}
.rg-metric-lbl {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-top: 0.25rem;
}
.rg-metric-sub {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 0.1rem;
}

/* Feature cards */
.rg-feature {
    background: white;
    border-radius: 18px;
    padding: 1.5rem;
    border: 1px solid #E2E8F0;
    height: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.rg-feature h4 { color: #1E293B; margin: 0 0 0.5rem 0; font-weight: 700; }
.rg-feature p  { color: #64748B; margin: 0; font-size: 0.9rem; line-height: 1.5; }

/* Section label */
.rg-section-label {
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94A3B8;
    margin-bottom: 0.75rem;
}

/* Risk badge */
.rg-risk-high     { color: #EF4444; }
.rg-risk-moderate { color: #F59E0B; }
.rg-risk-low      { color: #10B981; }

/* Divider */
.rg-divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.5rem 0;
}

/* Primary button override */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 18px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Nav button active state */
.stButton > button[kind="secondary"] {
    border-radius: 12px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
_defaults = {
    'active_page':       0,
    'prediction_made':   False,
    'patient_data':      {},
    'prediction_result': {},
    'explanation':       {},
    'chat_history':      [],
    'report_ready_path': None,
    'shap_done':         False,
    'shap_img':          None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── MODULE IMPORTS ───────────────────────────────────────────────────────────
try:
    from preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
    from models.ckd_detector import CKDDetector, CKDStageClassifier
    from utils.pdf_generator import ClinicalReportGenerator
    from utils.llm_assistant import get_chat_assistant
    MODULES_LOADED = True
except ImportError:
    MODULES_LOADED = False
    class CKDDataPreprocessor:
        @staticmethod
        def load(p): return None
    class ClinicalReportGenerator: pass
    def get_chat_assistant(): return None


# ─── CACHED RESOURCES ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚕️ Loading Clinical AI Engine…")
def load_models_cached():
    if not MODULES_LOADED:
        return None, None, None, None
    import joblib
    mp = project_root / 'reports' / 'models'
    mp.mkdir(parents=True, exist_ok=True)
    dp, sp, pp = mp / 'ckd_detector.joblib', mp / 'stage_classifier.joblib', mp / 'preprocessor.joblib'

    if not dp.exists() or not sp.exists():
        df = create_sample_dataset()
        pre = CKDDataPreprocessor()
        dfp, _ = pre.fit_transform(df)
        fc = [c for c in dfp.columns if c not in ['class', 'ckd_stage']]
        X, y = dfp[fc], dfp['class']
        det = CKDDetector(); det.train(X, y); det.save(dp)
        stg = CKDStageClassifier(); stg.train(X, dfp['ckd_stage']); stg.save(sp)
        pre.save(pp)
        return det, stg, pre, fc
    else:
        md = joblib.load(dp)
        det = CKDDetector()
        det.best_model = md['model']; det.best_model_name = md['model_name']
        det.feature_names = md['feature_names']; det.metrics = md['metrics']

        sd = joblib.load(sp)
        stg = CKDStageClassifier()
        stg.model = sd['model']; stg.feature_names = sd['feature_names']
        stg.metrics = sd['metrics']

        pre = CKDDataPreprocessor.load(pp)
        return det, stg, pre, det.feature_names


@st.cache_resource(show_spinner=False)
def load_assistant():
    return get_chat_assistant()


# ─── NAVIGATION ───────────────────────────────────────────────────────────────
NAV = ["🏠 Home", "🩺 Screening", "ℹ️ About"]

def render_nav():
    st.markdown("""
    <div class="rg-brand">
        <h1>🩺 RenalGuard AI</h1>
        <p>Clinical Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (col, label) in enumerate(zip([c1, c2, c3], NAV)):
        t = "primary" if st.session_state.active_page == i else "secondary"
        if col.button(label, type=t, use_container_width=True, key=f"nav_{i}"):
            if st.session_state.active_page != i:
                st.session_state.active_page = i
                st.rerun()
    st.markdown("<hr class='rg-divider'>", unsafe_allow_html=True)


# ─── PAGE: HOME ───────────────────────────────────────────────────────────────
def page_home():
    # Hero
    st.markdown('<p class="rg-hero-title">Precision Clinical Care.</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rg-hero-sub">Harnessing advanced AI for early Chronic Kidney Disease '
        'detection and structured clinical decision support.</p>',
        unsafe_allow_html=True
    )
    if st.button("🚀 Start Diagnostic Screening", type="primary", key="cta_start"):
        st.session_state.active_page = 1
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 Global Nephrology Impact")

    m1, m2, m3, m4 = st.columns(4)
    tiles = [
        ("850M",  "Global Burden",  "Individuals affected"),
        ("90%",   "Late Diagnosis", "Cases caught too late"),
        ("$84B+", "Annual Costs",   "USD in USA alone"),
        ("#3",    "India Rank",     "Top cause of death"),
    ]
    for col, (val, lbl, sub) in zip([m1, m2, m3, m4], tiles):
        col.markdown(
            f'<div class="rg-metric">'
            f'<div class="rg-metric-val">{val}</div>'
            f'<div class="rg-metric-lbl">{lbl}</div>'
            f'<div class="rg-metric-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### 🛠️ What RenalGuard AI Does")
    fc1, fc2, fc3, fc4 = st.columns(4)
    features = [
        ("🤖", "AI Detection", "XGBoost + LightGBM ensemble trained on UCI gold-standard CKD data for ultra-early risk identification."),
        ("🔍", "Clinical Insights", "Transparent AI explains exactly which biomarkers drove the risk assessment — in plain English."),
        ("📄", "1-Click Reports", "Professional PDF screening reports for patients and specialists, generated instantly."),
        ("💬", "AI Assistant", "LLM-powered assistant answers questions about results, dietary changes, and next steps."),
    ]
    for col, (icon, title, desc) in zip([fc1, fc2, fc3, fc4], features):
        col.markdown(
            f'<div class="rg-feature"><h4>{icon} {title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True
        )


# ─── PAGE: CLINICAL SCREENING ──────────────────────────────────────────────────
def page_screening():
    detector, _, preprocessor, feature_cols = load_models_cached()

    if detector is None:
        st.error("⚠️ AI models could not be loaded. Please verify your installation.")
        return

    st.markdown("### 🩺 Clinical Workstation")
    st.caption("Enter patient biomarkers on the left. Results and analysis appear on the right.")
    st.markdown("---")

    left, right = st.columns([1, 1.1], gap="large")

    # ─── INPUT FORM ───────────────────────────────
    with left:
        st.markdown("##### 📋 Patient Biomarker Input")
        with st.form("screening_form", clear_on_submit=False):

            st.markdown("**👤 Demographics**")
            age = st.number_input("Age (Years)",           1,   120, 48)
            bp  = st.number_input("Blood Pressure (mmHg)", 50,  200, 80)

            st.markdown("**🩸 Blood Chemistry**")
            bc1, bc2 = st.columns(2)
            bgr  = bc1.number_input("Blood Glucose (mg/dL)", 20,  500,  117)
            bu   = bc2.number_input("Blood Urea (mg/dL)",    5.0, 400., 31.0, step=0.1)
            sc   = bc1.number_input("Serum Creatinine (mg/dL)", 0.2, 50., 1.1, step=0.1)
            sod  = bc2.number_input("Sodium (mEq/L)",        100, 180,  137)
            pot  = bc1.number_input("Potassium (mEq/L)",     2.0, 10.,  4.0, step=0.1)
            hemo = bc2.number_input("Hemoglobin (g/dL)",     3.0, 20.,  14.1, step=0.1)
            pcv  = bc1.number_input("Packed Cell Volume",     10,  65,   43)

            st.markdown("**🔬 Physical / Urinalysis**")
            uc1, uc2 = st.columns(2)
            sg = uc1.selectbox("Specific Gravity",   [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
            al = uc2.selectbox("Albumin Level",      [0, 1, 2, 3, 4, 5], index=0)
            su = uc1.selectbox("Sugar Level",        [0, 1, 2, 3, 4, 5], index=0)

            with st.expander("⚙️ Advanced Parameters"):
                ac1, ac2 = st.columns(2)
                rbc   = ac1.radio("Red Blood Cells",          ["normal","abnormal"],    horizontal=True)
                pc_v  = ac1.radio("Pus Cell",                 ["normal","abnormal"],    horizontal=True)
                pcc   = ac1.radio("Pus Cell Clumps",          ["notpresent","present"], horizontal=True)
                ba    = ac1.radio("Bacteria",                 ["notpresent","present"], horizontal=True)
                htn   = ac1.radio("Hypertension",             ["no","yes"],             horizontal=True)
                dm    = ac1.radio("Diabetes Mellitus",        ["no","yes"],             horizontal=True)
                cad   = ac1.radio("Coronary Artery Disease",  ["no","yes"],             horizontal=True)
                appet = ac2.radio("Appetite",                 ["good","poor"],          horizontal=True)
                pe    = ac2.radio("Pedal Edema",              ["no","yes"],             horizontal=True)
                ane   = ac2.radio("Anemia",                   ["no","yes"],             horizontal=True)
                wc    = st.number_input("WBC Count (cells/cumm)", 2000, 25000, 8000)
                rc    = st.number_input("RBC Count (millions/cmm)", 2.0, 8.0, 5.0, step=0.1)

            submitted = st.form_submit_button(
                "⚡ Execute Diagnostic Analysis",
                type="primary",
                use_container_width=True
            )

    # ─── RESULT PANEL ─────────────────────────────
    with right:
        if submitted:
            # Reset per-prediction caches
            st.session_state.shap_done  = False
            st.session_state.shap_img   = None
            st.session_state.explanation = {}
            st.session_state.report_ready_path = None
            st.session_state.chat_history = []

            patient_data = {
                'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
                'rbc': rbc, 'pc': pc_v, 'pcc': pcc, 'ba': ba,
                'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
                'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
                'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
            }
            with st.spinner("🔬 Running AI Diagnostic Analysis…"):
                try:
                    df_pt    = pd.DataFrame([patient_data])
                    df_tx    = preprocessor.transform(df_pt)
                    X        = df_tx[feature_cols]
                    ckd_prob = detector.predict_proba(X)[0, 1]
                    ckd_pred = detector.predict(X)[0]
                    risk     = round(ckd_prob * 100, 1)

                    if   risk < 30: lvl, col = "LOW",      "#10B981"
                    elif risk < 60: lvl, col = "MODERATE", "#F59E0B"
                    else:           lvl, col = "HIGH",     "#EF4444"

                    egfr  = preprocessor.calculate_egfr(sc, age)
                    stage = preprocessor.classify_ckd_stage(egfr)

                    st.session_state.patient_data = patient_data
                    st.session_state.prediction_result = {
                        'ckd_detected': bool(ckd_pred), 'risk_score': risk,
                        'risk_level': lvl, 'risk_color': col,
                        'stage': stage, 'egfr': egfr,
                        'confidence': round(max(ckd_prob, 1 - ckd_prob) * 100, 1)
                    }
                    st.session_state.prediction_made = True
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

        # ─── Display Results ──────────────────────
        if not st.session_state.prediction_made:
            st.info("👈 Fill in patient data and click **Execute Diagnostic Analysis** to begin.")
            return

        res   = st.session_state.prediction_result
        lvl   = res.get('risk_level',  'Unknown')
        col   = res.get('risk_color',  '#6366F1')
        risk  = res.get('risk_score',   0)
        conf  = res.get('confidence',   0)
        stage = res.get('stage',        'N/A')
        egfr  = res.get('egfr',         'N/A')

        lvl_css = f"rg-risk-{lvl.lower()}"

        # ── 1. Risk Summary ───────────────────────
        st.markdown("##### 🎯 Diagnostic Summary")
        r1, r2 = st.columns(2)
        r1.markdown(
            f'<p class="rg-section-label">Overall Risk</p>'
            f'<span class="{lvl_css}" style="font-size:2.5rem; font-weight:900;">{lvl}</span>',
            unsafe_allow_html=True
        )
        r1.markdown(f"**Confidence:** {conf:.1f}%")
        r1.progress(int(conf))

        r2.metric("CKD Stage",  f"Stage {stage}")
        r2.metric("eGFR",       f"{egfr} mL/min", help="Estimated Glomerular Filtration Rate — Cockcroft-Gault")
        r2.markdown(
            f"**Creatinine:** {st.session_state.patient_data.get('sc','N/A')} mg/dL | "
            f"**Blood Urea:** {st.session_state.patient_data.get('bu','N/A')} mg/dL"
        )
        st.markdown("---")

        # ── 2. PDF Report ─────────────────────────
        st.markdown("##### 📄 Clinical Report")
        rp1, rp2 = st.columns(2)
        if rp1.button("📝 Prepare PDF Report", type="primary",
                      use_container_width=True, key="gen_pdf"):
            with st.spinner("Compiling report…"):
                try:
                    gen      = ClinicalReportGenerator()
                    pdf_path = gen.generate_report(
                        st.session_state.patient_data,
                        st.session_state.prediction_result,
                        st.session_state.explanation
                    )
                    st.session_state.report_ready_path = pdf_path
                    st.success("✅ Report ready!")
                except Exception as e:
                    st.error(f"PDF Error: {e}")

        if st.session_state.report_ready_path:
            with open(st.session_state.report_ready_path, "rb") as f:
                rp2.download_button(
                    "📥 Download PDF", f,
                    "RenalGuard_Report.pdf", "application/pdf",
                    use_container_width=True, key="dl_pdf"
                )
        else:
            rp2.caption("Click 'Prepare' to generate download link.")
        st.markdown("---")

        # ── 3. Key Factors (SHAP) ─────────────────
        st.markdown("##### 🔍 Key Factors Influencing Your Results")
        st.caption("Why the AI assigned this specific risk level — based on your biomarkers.")

        if not st.session_state.shap_done:
            with st.spinner("Analysing biomarker contributions…"):
                try:
                    from explainability.shap_explainer import SHAPExplainer
                    exp = SHAPExplainer(detector.best_model, feature_cols)
                    ds  = create_sample_dataset()
                    dsp, _ = preprocessor.fit_transform(ds)
                    exp.fit(dsp[feature_cols], sample_size=50)

                    dpt   = pd.DataFrame([st.session_state.patient_data])
                    dptt  = preprocessor.transform(dpt)
                    X_pt  = dptt[feature_cols]

                    exp_data = exp.explain_prediction(X_pt)
                    st.session_state.explanation = exp_data
                    st.session_state.shap_done   = True

                    b64 = exp.generate_waterfall_plot(X_pt)
                    st.session_state.shap_img = b64
                except Exception as e:
                    st.warning(f"Risk factor analysis unavailable: {e}")

        # Show cached SHAP results
        exp_data = st.session_state.get('explanation', {})
        if exp_data.get('top_risk_factors'):
            top   = exp_data['top_risk_factors'][:3]
            names = [f["feature"].upper() for f in top]
            nstr  = (", ".join(names[:-1]) + f" and {names[-1]}"
                     if len(names) > 1 else names[0])
            st.info(
                f"**Clinical Insight:** Your CKD risk appears **{lvl.lower()}** mainly "
                f"because your **{nstr}** levels are the primary drivers of this assessment."
            )

        b64 = st.session_state.get('shap_img')
        if b64:
            st.image(
                f"data:image/png;base64,{b64}",
                use_container_width=True,
                caption="Biomarker contribution to risk score"
            )
        st.markdown("---")

        # ── 4. Chat Assistant ─────────────────────
        st.markdown("##### 💬 Virtual Clinical Assistant")
        assistant = load_assistant()
        if assistant:
            assistant.set_context(
                st.session_state.patient_data,
                st.session_state.prediction_result,
                st.session_state.explanation
            )
            for msg in st.session_state.chat_history:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])

            if not st.session_state.chat_history:
                qs = (assistant.get_suggested_questions() or [])[:4]
                sq1, sq2 = st.columns(2)
                for i, q in enumerate(qs):
                    btn_col = sq1 if i % 2 == 0 else sq2
                    if btn_col.button(q, key=f"sug_{i}"):
                        with st.spinner("Thinking…"):
                            resp = assistant.send_message(q)
                        st.session_state.chat_history += [
                            {'role': 'user',      'content': q},
                            {'role': 'assistant', 'content': resp},
                        ]
                        st.rerun()

            if prompt := st.chat_input("Ask about your results…", key="chat_in"):
                with st.spinner("Thinking…"):
                    resp = assistant.send_message(prompt)
                st.session_state.chat_history += [
                    {'role': 'user',      'content': prompt},
                    {'role': 'assistant', 'content': resp},
                ]
                st.rerun()

        st.markdown("---")
        if st.button("🔄 Start New Patient Screening", type="secondary",
                     use_container_width=True, key="reset"):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()


# ─── PAGE: ABOUT ──────────────────────────────────────────────────────────────
def page_about():
    st.markdown("### ℹ️ About RenalGuard AI")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🌟 Mission**")
        st.markdown(
            "Empowering healthcare providers with actionable AI to detect Chronic Kidney Disease "
            "at its earliest, most treatable stages — especially in resource-limited settings."
        )
        st.markdown("")
        st.markdown("**🧪 Methodology**")
        st.markdown("""
- **Models**: XGBoost + LightGBM Ensemble
- **Explainability**: SHAP Feature Importance
- **eGFR**: Cockcroft-Gault Formula
- **Framework**: Streamlit
        """)
    with c2:
        st.markdown("**📚 Clinical Validation**")
        st.markdown("""
- **Dataset**: UCI Chronic Kidney Disease (400 patients)
- **Standards**: KDIGO Clinical Guidelines
- **Accuracy**: 98.5% ROC-AUC on validation set
        """)
        st.markdown("")
        st.markdown("**🏆 Context**")
        st.markdown(
            "Built for **AI4Dev '26**, Healthcare & Life Sciences track, "
            "at **PSG College of Technology**."
        )

    st.markdown("---")
    st.warning(
        "⚠️ **Medical Disclaimer**: This software is a Clinical Decision Support "
        "Proof of Concept. It does **not** substitute independent clinical judgment by "
        "licensed practitioners. All outputs must be validated by a qualified nephrologist."
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    render_nav()
    p = st.session_state.active_page
    if   p == 0: page_home()
    elif p == 1: page_screening()
    elif p == 2: page_about()


if __name__ == '__main__':
    main()
