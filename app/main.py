"""
RenalGuard AI - Main Streamlit Application
AI-Powered Early Detection & Clinical Decision Support for Chronic Kidney Disease
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RenalGuard AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
css_path = Path(__file__).parent / 'style.css'
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE  (initialise once)
# ─────────────────────────────────────────────
_defaults = {
    'active_page': 0,          # 0=Home, 1=Clinical Suite, 2=About
    'prediction_made': False,
    'patient_data': {},
    'prediction_result': {},
    'explanation': {},
    'chat_history': [],
    'report_ready_path': None,
    'shap_done': False,        # flag: SHAP computed for current prediction
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# MODULE IMPORTS  (graceful fallback)
# ─────────────────────────────────────────────
try:
    from preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
    from models.ckd_detector import CKDDetector, CKDStageClassifier
    from utils.pdf_generator import ClinicalReportGenerator
    from utils.llm_assistant import get_chat_assistant
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    # Stub classes so the rest of the code doesn't break
    class CKDDataPreprocessor:
        @staticmethod
        def load(p): return None
    class ClinicalReportGenerator: pass
    def get_chat_assistant(): return None


# ─────────────────────────────────────────────
# CACHED MODEL LOADING  (loaded once per session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⚕️ Initialising Clinical Intelligence Engine…")
def load_models_cached():
    """Load or train models — runs only ONCE and is cached for the entire session."""
    if not MODULES_LOADED:
        return None, None, None, None

    models_path = project_root / 'reports' / 'models'
    models_path.mkdir(parents=True, exist_ok=True)

    detector_path   = models_path / 'ckd_detector.joblib'
    stage_path      = models_path / 'stage_classifier.joblib'
    preproc_path    = models_path / 'preprocessor.joblib'
    import joblib

    if not detector_path.exists() or not stage_path.exists():
        df = create_sample_dataset()
        preprocessor = CKDDataPreprocessor()
        df_processed, _ = preprocessor.fit_transform(df)
        feature_cols = [c for c in df_processed.columns if c not in ['class', 'ckd_stage']]
        X, y = df_processed[feature_cols], df_processed['class']

        detector = CKDDetector()
        detector.train(X, y)
        detector.save(detector_path)

        stage_clf = CKDStageClassifier()
        stage_clf.train(X, df_processed['ckd_stage'])
        stage_clf.save(stage_path)

        preprocessor.save(preproc_path)
        return detector, stage_clf, preprocessor, feature_cols
    else:
        model_data = joblib.load(detector_path)
        detector = CKDDetector()
        detector.best_model      = model_data['model']
        detector.best_model_name = model_data['model_name']
        detector.feature_names   = model_data['feature_names']
        detector.metrics         = model_data['metrics']

        stage_data = joblib.load(stage_path)
        stage_clf = CKDStageClassifier()
        stage_clf.model         = stage_data['model']
        stage_clf.feature_names = stage_data['feature_names']
        stage_clf.metrics       = stage_data['metrics']

        preprocessor = CKDDataPreprocessor.load(preproc_path)
        return detector, stage_clf, preprocessor, detector.feature_names


@st.cache_resource(show_spinner=False)
def load_chat_assistant():
    return get_chat_assistant()


# ─────────────────────────────────────────────
# NAVIGATION HELPERS
# ─────────────────────────────────────────────
NAV_LABELS = ["🏠 Home", "🩺 Clinical Suite", "ℹ️ About"]

def go_to(page_idx: int):
    st.session_state.active_page = page_idx
    st.rerun()


def render_nav():
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size:1.5rem; font-weight:900; color:#6366F1;">🩺 RenalGuard AI</span><br>
        <span style="color:#64748B; font-size:0.85rem;">Clinical Decision Support System</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(NAV_LABELS))
    for i, (col, label) in enumerate(zip(cols, NAV_LABELS)):
        with col:
            is_active = st.session_state.active_page == i
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{i}", type=btn_type, use_container_width=True):
                if not is_active:
                    st.session_state.active_page = i
                    st.rerun()
    st.markdown("---")


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
def show_home():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Precision Clinical Care.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Harnessing advanced AI for early Chronic Kidney Disease '
        'detection and structured clinical decision support.</p>',
        unsafe_allow_html=True
    )

    if st.button("🚀 Start Diagnostic Screening", type="primary", key="home_cta"):
        st.session_state.active_page = 1
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Global Nephrology Impact")
    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        ("Global Burden",   "850M",  "Individuals Affected"),
        ("Late Diagnosis",  "90%",   "Are Unaware"),
        ("Annual Costs",    "$84B+", "USD in USA alone"),
        ("India Rank",      "#3",    "Top cause of death"),
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4], tiles):
        col.markdown(
            f'<div class="metric-tile"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{sub}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Integrated Diagnostic Suite")
    fc1, fc2 = st.columns(2)
    cards = [
        ("🤖 AI Detection Engine",
         "Ensemble models (XGBoost + LightGBM) trained on gold-standard UCI clinical data for ultra-early risk identification."),
        ("📄 Clinical Reporting",
         "Generate professional-grade PDF reports for patients and specialists with automated clinical narratives."),
        ("🔍 Clinical Insights",
         "Transparent AI explains which biomarkers (like Creatinine and Hemoglobin) drove the risk assessment in plain English."),
        ("💬 Virtual Clinical Assistant",
         "LLM-integrated interface to help clinicians and patients understand results immediately after screening."),
    ]
    for i, (title, desc) in enumerate(cards):
        col = fc1 if i % 2 == 0 else fc2
        col.markdown(
            f'<div class="glass-card"><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: CLINICAL SUITE
# ─────────────────────────────────────────────
def show_clinical_suite():
    detector, stage_clf, preprocessor, feature_cols = load_models_cached()

    if detector is None:
        st.error("⚠️ Clinical models could not be loaded. Please check requirements.")
        return

    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title" style="font-size:2.6rem;">🩺 Clinical Workstation</h1>',
                unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.2], gap="large")

    # ── INPUT FORM ────────────────────────────
    with col_input:
        with st.form("screening_form", clear_on_submit=False):
            st.markdown("#### 👤 Patient Demographics")
            age = st.number_input("Age (Years)",        1,  120, 48)
            bp  = st.number_input("Blood Pressure (mmHg)", 50, 200, 80)

            st.markdown("#### 🩸 Blood Chemistry")
            cc1, cc2 = st.columns(2)
            with cc1:
                bgr  = st.number_input("Blood Glucose (mg/dL)", 20,  500,  117)
                bu   = st.number_input("Blood Urea (mg/dL)",    5.0, 400., 31.0, step=0.1)
                sc   = st.number_input("Serum Creatinine (mg/dL)", 0.2, 50., 1.1, step=0.1)
            with cc2:
                sod  = st.number_input("Sodium (mEq/L)",       100, 180,  137)
                pot  = st.number_input("Potassium (mEq/L)",    2.0, 10.,  4.0, step=0.1)
                hemo = st.number_input("Hemoglobin (g/dL)",    3.0, 20.,  14.1, step=0.1)
                pcv  = st.number_input("Packed Cell Volume",    10,  65,   43)

            st.markdown("#### 🔬 Physical Attributes")
            pc1, pc2 = st.columns(2)
            with pc1:
                sg = st.selectbox("Specific Gravity",
                                  [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
                al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5], index=0)
            with pc2:
                su = st.selectbox("Sugar",   [0, 1, 2, 3, 4, 5], index=0)

            with st.expander("⚙️ Advanced Clinical Parameters"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    rbc  = st.radio("Red Blood Cells",         ["normal","abnormal"],    horizontal=True)
                    pc_v = st.radio("Pus Cell",                ["normal","abnormal"],    horizontal=True)
                    pcc  = st.radio("Pus Cell Clumps",         ["notpresent","present"], horizontal=True)
                    ba   = st.radio("Bacteria",                ["notpresent","present"], horizontal=True)
                    htn  = st.radio("Hypertension",            ["no","yes"],             horizontal=True)
                    dm   = st.radio("Diabetes Mellitus",       ["no","yes"],             horizontal=True)
                    cad  = st.radio("Coronary Artery Disease", ["no","yes"],             horizontal=True)
                with ec2:
                    appet = st.radio("Appetite",    ["good","poor"], horizontal=True)
                    pe    = st.radio("Pedal Edema", ["no","yes"],    horizontal=True)
                    ane   = st.radio("Anemia",      ["no","yes"],    horizontal=True)
                wc = st.number_input("WBC Count", 2000, 25000, 8000)
                rc = st.number_input("RBC Count", 2.0,   8.0,  5.0, step=0.1)

            submitted = st.form_submit_button(
                "⚡ Execute Diagnostic Analysis",
                type="primary",
                use_container_width=True
            )

    # ── RESULTS PANEL ─────────────────────────
    with col_result:
        if submitted:
            # Reset SHAP cache when new form is submitted
            st.session_state.shap_done = False
            st.session_state.explanation = {}
            st.session_state.report_ready_path = None

            patient_data = {
                'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
                'rbc': rbc, 'pc': pc_v, 'pcc': pcc, 'ba': ba,
                'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
                'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
                'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
            }
            try:
                df_patient   = pd.DataFrame([patient_data])
                df_transformed = preprocessor.transform(df_patient)
                X            = df_transformed[feature_cols]

                ckd_prob  = detector.predict_proba(X)[0, 1]
                ckd_pred  = detector.predict(X)[0]
                risk_score = round(ckd_prob * 100, 1)

                if   risk_score < 30: risk_level, risk_color = "LOW",      "#10B981"
                elif risk_score < 60: risk_level, risk_color = "MODERATE", "#F59E0B"
                else:                 risk_level, risk_color = "HIGH",     "#EF4444"

                egfr  = preprocessor.calculate_egfr(sc, age)
                stage = preprocessor.classify_ckd_stage(egfr)

                st.session_state.patient_data      = patient_data
                st.session_state.prediction_result = {
                    'ckd_detected': bool(ckd_pred),
                    'risk_score':   risk_score,
                    'risk_level':   risk_level,
                    'risk_color':   risk_color,
                    'stage':        stage,
                    'egfr':         egfr,
                    'confidence':   round(max(ckd_prob, 1 - ckd_prob) * 100, 1)
                }
                st.session_state.prediction_made = True
            except Exception as e:
                st.error(f"Prediction Error: {e}")

        # ── Show stored result ─────────────────
        if st.session_state.prediction_made:
            res        = st.session_state.prediction_result
            risk_level = res.get('risk_level', 'Unknown')
            risk_color = res.get('risk_color', '#6366F1')
            risk_score = res.get('risk_score', 0)
            confidence = res.get('confidence', 0)
            res_class  = ("risk-success" if risk_level == "LOW"
                          else "risk-warning" if risk_level == "MODERATE"
                          else "risk-danger")

            # ── Diagnostic Result Card ──────────
            st.markdown(f'<div class="glass-card report-card {res_class}">', unsafe_allow_html=True)
            r1, r2 = st.columns([1, 1.5])
            with r1:
                st.markdown("**Diagnostic Summary**")
                st.markdown(
                    f"<h1 style='color:{risk_color}; margin:0; line-height:1;'>{risk_level}</h1>"
                    f"<p style='color:{risk_color}; font-weight:700; margin:0;'>CKD RISK</p>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**AI Confidence**: `{confidence:.1f}%`")
                st.progress(int(confidence))
                st.caption(f"Risk Score: {risk_score:.1f} / 100")
            with r2:
                st.markdown("**Clinical Parameters**")
                m1, m2 = st.columns(2)
                m1.metric("Predicted Stage", f"Stage {res.get('stage', 'N/A')}")
                m2.metric("eGFR", f"{res.get('egfr', 'N/A')} mL/min")
                pd_data = st.session_state.patient_data
                st.markdown(
                    f"**Creatinine**: {pd_data.get('sc','N/A')} mg/dL &nbsp;|&nbsp; "
                    f"**Blood Urea**: {pd_data.get('bu','N/A')} mg/dL",
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Report Generation ───────────────
            st.markdown("#### 📄 Clinical Report")
            rp1, rp2 = st.columns(2)
            with rp1:
                if st.button("📝 Prepare Digital Report", type="primary",
                             use_container_width=True, key="gen_pdf"):
                    with st.spinner("Compiling report…"):
                        try:
                            pdf_gen  = ClinicalReportGenerator()
                            pdf_path = pdf_gen.generate_report(
                                st.session_state.patient_data,
                                st.session_state.prediction_result,
                                st.session_state.explanation
                            )
                            st.session_state.report_ready_path = pdf_path
                            st.success("Report ready!")
                        except Exception as e:
                            st.error(f"PDF Error: {e}")
            with rp2:
                if st.session_state.report_ready_path:
                    with open(st.session_state.report_ready_path, "rb") as f:
                        st.download_button(
                            "📥 Download PDF",
                            f,
                            "RenalGuard_Clinical_Report.pdf",
                            "application/pdf",
                            use_container_width=True,
                            key="dl_pdf"
                        )
                else:
                    st.info("Click 'Prepare' first.")

            st.divider()

            # ── Key Factors (SHAP) ──────────────
            st.markdown("#### 🔍 Key Factors Influencing Your Results")
            st.caption("Why the AI assigned this specific risk level based on your biomarkers.")

            # Only compute SHAP once per prediction
            if not st.session_state.shap_done:
                try:
                    from explainability.shap_explainer import SHAPExplainer
                    with st.spinner("Analysing biomarker contributions…"):
                        explainer = SHAPExplainer(detector.best_model, feature_cols)
                        df_s      = create_sample_dataset()
                        df_p, _   = preprocessor.fit_transform(df_s)
                        X_s       = df_p[feature_cols]
                        explainer.fit(X_s, sample_size=50)

                        df_pt   = pd.DataFrame([st.session_state.patient_data])
                        df_pt_t = preprocessor.transform(df_pt)
                        X_pt    = df_pt_t[feature_cols]

                        exp_data = explainer.explain_prediction(X_pt)
                        st.session_state.explanation = exp_data
                        st.session_state.shap_done   = True

                        img_b64 = explainer.generate_waterfall_plot(X_pt)
                        if img_b64:
                            st.session_state._shap_img = img_b64
                except Exception as e:
                    st.warning(f"Clinical insight unavailable: {e}")

            # Show cached SHAP results
            exp_data = st.session_state.get('explanation', {})
            if exp_data.get('top_risk_factors'):
                top      = exp_data['top_risk_factors'][:3]
                names    = [f["feature"].upper() for f in top]
                name_str = (", ".join(names[:-1]) + f" and {names[-1]}"
                            if len(names) > 1 else names[0])
                st.info(
                    f"**Clinical Insight:** Your CKD risk appears **{risk_level.lower()}** "
                    f"mainly because your **{name_str}** levels are the primary drivers."
                )

            img_b64 = st.session_state.get('_shap_img')
            if img_b64:
                st.image(
                    f"data:image/png;base64,{img_b64}",
                    width=None,
                    caption="Biomarker contribution breakdown"
                )

            st.divider()

            # ── AI Chat Assistant ───────────────
            st.markdown("#### 💬 Virtual Clinical Assistant")
            assistant = load_chat_assistant()
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
                    suggestions = assistant.get_suggested_questions()[:4]
                    sq1, sq2 = st.columns(2)
                    for i, q in enumerate(suggestions):
                        with (sq1 if i % 2 == 0 else sq2):
                            if st.button(q, key=f"sug_{i}"):
                                with st.spinner("Thinking…"):
                                    resp = assistant.send_message(q)
                                st.session_state.chat_history += [
                                    {'role': 'user',      'content': q},
                                    {'role': 'assistant', 'content': resp}
                                ]
                                st.rerun()

                if prompt := st.chat_input("Ask about your results…", key="chat_input"):
                    with st.spinner("Thinking…"):
                        resp = assistant.send_message(prompt)
                    st.session_state.chat_history += [
                        {'role': 'user',      'content': prompt},
                        {'role': 'assistant', 'content': resp}
                    ]
                    st.rerun()

            st.divider()
            if st.button("🔄 Start New Patient Screening", type="secondary",
                         use_container_width=True, key="reset"):
                for k in ['prediction_made', 'patient_data', 'prediction_result',
                          'explanation', 'chat_history', 'report_ready_path',
                          'shap_done', '_shap_img']:
                    st.session_state[k] = _defaults.get(k, None if k != 'chat_history'
                                                         else [])
                st.rerun()

        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; color:#94A3B8; padding:4rem 1rem;">
                <div style="font-size:4rem; opacity:0.2;">📋</div>
                <p style="margin-top:1rem;">
                    Enter patient biomarkers on the left<br>and click <strong>Execute Diagnostic Analysis</strong>.
                </p>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
def show_about():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("## ℹ️ About RenalGuard AI")

    st.markdown("""
    <div class="glass-card">
        <h3>🌟 Mission</h3>
        <p>Empowering healthcare providers with actionable AI to detect Chronic Kidney Disease 
        at its earliest, most treatable stages — especially in resource-limited settings.</p>
        <h3>🏆 Context</h3>
        <p>Developed for <strong>AI4Dev '26</strong>, Healthcare & Life Sciences track, 
        at <strong>PSG College of Technology</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    ab1, ab2 = st.columns(2)
    ab1.markdown("""
    <div class="glass-card">
        <h4>🧪 Methodology</h4>
        <ul>
            <li><strong>Models</strong>: XGBoost + LightGBM ensemble</li>
            <li><strong>XAI</strong>: SHAP feature importance</li>
            <li><strong>Framework</strong>: Streamlit</li>
        </ul>
    </div>""", unsafe_allow_html=True)
    ab2.markdown("""
    <div class="glass-card">
        <h4>📚 Clinical Validation</h4>
        <ul>
            <li><strong>Dataset</strong>: UCI CKD Dataset (400 patients)</li>
            <li><strong>Standards</strong>: KDIGO Clinical Guidelines</li>
            <li><strong>eGFR</strong>: Cockcroft-Gault Formula</li>
        </ul>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="border-left:5px solid #EF4444;">
        <h4>⚠️ Medical Disclaimer</h4>
        <p>This software is a <strong>Clinical Decision Support Proof of Concept</strong>. 
        It does <strong>not</strong> substitute independent clinical judgment by licensed practitioners. 
        All outputs must be validated by a qualified nephrologist.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    render_nav()
    page = st.session_state.active_page
    if   page == 0: show_home()
    elif page == 1: show_clinical_suite()
    elif page == 2: show_about()


if __name__ == '__main__':
    main()
