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

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Configure page
st.set_page_config(
    page_title="RenalGuard AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Path to CSS
css_path = Path(__file__).parent / 'style.css'
if css_path.exists():
    local_css(str(css_path))
else:
    # Fallback if file not found
    st.markdown("""
    <style>
        .hero-title { font-weight: 900; font-size: 4rem; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = {}
if 'explanation' not in st.session_state:
    st.session_state.explanation = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home Dashboard"

# Import modules
try:
    from preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
    from models.ckd_detector import CKDDetector, CKDStageClassifier
    from utils.pdf_generator import ClinicalReportGenerator
    from utils.llm_assistant import get_chat_assistant
    MODULES_LOADED = True
except ImportError as e:
    critical_missing = ["preprocessing", "models", "explainability"]
    is_critical = any(m in str(e) for m in critical_missing)
    if is_critical:
        st.error(f"Critical Module Error: {e}")
    else:
        pass  # Non-critical 
    
    if 'CKDDataPreprocessor' not in locals():
        class CKDDataPreprocessor: 
            @staticmethod
            def load(p): pass
    if 'ClinicalReportGenerator' not in locals():
        class ClinicalReportGenerator: pass
    if 'get_chat_assistant' not in locals():
        def get_chat_assistant(): return None
    
    MODULES_LOADED = False


def load_models():
    models_path = Path(__file__).parent.parent / 'reports' / 'models'
    models_path.mkdir(parents=True, exist_ok=True)
    
    detector_path = models_path / 'ckd_detector.joblib'
    stage_path = models_path / 'stage_classifier.joblib'
    
    if not detector_path.exists() or not stage_path.exists():
        with st.spinner('Initializing Clinical Models...'):
            df = create_sample_dataset()
            preprocessor = CKDDataPreprocessor()
            df_processed, _ = preprocessor.fit_transform(df)
            feature_cols = [col for col in df_processed.columns if col not in ['class', 'ckd_stage']]
            X = df_processed[feature_cols]
            y = df_processed['class']
            
            detector = CKDDetector()
            detector.train(X, y)
            detector.save(detector_path)
            
            stage_clf = CKDStageClassifier()
            stage_clf.train(X, df_processed['ckd_stage'])
            stage_clf.save(stage_path)
            
            preprocessor.save(models_path / 'preprocessor.joblib')
            return detector, stage_clf, preprocessor, feature_cols
    else:
        model_data = __import__('joblib').load(detector_path)
        detector = CKDDetector()
        detector.best_model = model_data['model']
        detector.best_model_name = model_data['model_name']
        detector.feature_names = model_data['feature_names']
        detector.metrics = model_data['metrics']
        
        stage_data = __import__('joblib').load(stage_path)
        stage_clf = CKDStageClassifier()
        stage_clf.model = stage_data['model']
        stage_clf.feature_names = stage_data['feature_names']
        stage_clf.metrics = stage_data['metrics']
        
        preprocessor = CKDDataPreprocessor.load(models_path / 'preprocessor.joblib')
        feature_cols = detector.feature_names
        
        return detector, stage_clf, preprocessor, feature_cols


def main():
    # Load Models
    if 'models_loaded' not in st.session_state:
        if MODULES_LOADED:
            try:
                st.session_state.detector, st.session_state.stage_clf, st.session_state.preprocessor, st.session_state.feature_cols = load_models()
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.session_state.models_loaded = False
        else:
            st.session_state.models_loaded = False

    # Top Navigation Bar
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding-top: 1rem; padding-bottom: 2rem;">
        <h2 style="color: var(--primary); font-weight: 800; margin-bottom: 0;">🩺 RenalGuard AI</h2>
        <p style="color: #64748B;">Clinical Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)

    # Simplified Navigation
    st.radio(
        "Navigation",
        ["🏠 Home Dashboard", "🩺 Clinical Suite", "ℹ️ About Project"],
        horizontal=True,
        label_visibility="collapsed",
        key="page"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('---')

    if st.session_state.page == "🏠 Home Dashboard":
        show_home_page()
    elif st.session_state.page == "🩺 Clinical Suite":
        show_clinical_suite()
    elif st.session_state.page == "ℹ️ About Project":
        show_about_page()


def show_home_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Precision Clinical Care.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Harnessing advanced AI for early Chronic Kidney Disease detection and structured clinical decision support.</p>', unsafe_allow_html=True)
    
    if st.button("🚀 Start Diagnostic Screening", type="primary"):
        st.session_state.page = "🩺 Clinical Suite"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Global Nephrology Impact")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('''<div class="metric-tile"><div class="metric-label">Global Burden</div><div class="metric-value">850M</div><div class="metric-label">Individuals Affected</div></div>''', unsafe_allow_html=True)
    with col2:
        st.markdown('''<div class="metric-tile"><div class="metric-label">Late Diagnosis</div><div class="metric-value">90%</div><div class="metric-label">Are Unaware</div></div>''', unsafe_allow_html=True)
    with col3:
        st.markdown('''<div class="metric-tile"><div class="metric-label">Annual Costs</div><div class="metric-value">84B+</div><div class="metric-label">USD in USA alone</div></div>''', unsafe_allow_html=True)
    with col4:
        st.markdown('''<div class="metric-tile"><div class="metric-label">India Rank</div><div class="metric-value">#3</div><div class="metric-label">Top cause of death</div></div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Integrated Diagnostic Suite")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown('''<div class="glass-card"><h4>🤖 AI Detection Engine</h4><p>Ensemble models (XGBoost + LightGBM) trained on gold-standard clinical data for ultra-early risk identification.</p></div>''', unsafe_allow_html=True)
        st.markdown('''<div class="glass-card"><h4>📄 Clinical Reporting</h4><p>Generate professional-grade PDF reports for patients and specialists with automated clinical narratives.</p></div>''', unsafe_allow_html=True)
    with f_col2:
        st.markdown('''<div class="glass-card"><h4>🔍 SHAP Interpretability</h4><p>Transparent AI that explains 'why' a risk level was assigned, visualizing key contributors like Creatinine and Hemoglobin.</p></div>''', unsafe_allow_html=True)
        st.markdown('''<div class="glass-card"><h4>💬 Virtual Clinical Assistant</h4><p>LLM-integrated consultative interface to help clinicians and patients respond to screening actions immediately.</p></div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_clinical_suite():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title" style="font-size: 3rem;">🩺 Clinical Workstation</h1>', unsafe_allow_html=True)
    
    if not st.session_state.get('models_loaded'):
        st.error("System modules offline. Cannot perform screening.")
        return

    detector = st.session_state.detector
    preprocessor = st.session_state.preprocessor
    feature_cols = st.session_state.feature_cols

    col_input, col_result = st.columns([1, 1.2])
    
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Patient Biomarker Input")
        with st.form("screening_form"):
            st.markdown("#### Patient Demographics")
            age = st.number_input("Age (Years)", 1, 120, 48)
            bp = st.number_input("Blood Pressure (mmHg)", 50, 200, 80)
            
            st.markdown("#### Blood Chemistry")
            col1, col2 = st.columns(2)
            with col1:
                bgr = st.number_input("Blood Glucose Random (mg/dL)", 20, 500, 117)
                bu = st.number_input("Blood Urea (mg/dL)", 5.0, 400.0, 31.0, step=0.1)
                sc = st.number_input("Serum Creatinine (mg/dL)", 0.2, 50.0, 1.1, step=0.1)
            with col2:
                sod = st.number_input("Sodium (mEq/L)", 100, 180, 137)
                pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.0, step=0.1)
                hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, 14.1, step=0.1)
                pcv = st.number_input("Packed Cell Volume", 10, 65, 43)

            st.markdown("#### Physical Attributes")
            col1, col2 = st.columns(2)
            with col1:
                sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
                al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5], index=0)
            with col2:
                su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5], index=0)

            with st.expander("Advanced Clinical Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    rbc = st.radio("Red Blood Cells", ["normal", "abnormal"], horizontal=True)
                    pc = st.radio("Pus Cell", ["normal", "abnormal"], horizontal=True)
                    pcc = st.radio("Pus Cell Clumps", ["notpresent", "present"], horizontal=True)
                    ba = st.radio("Bacteria", ["notpresent", "present"], horizontal=True)
                    htn = st.radio("Hypertension", ["no", "yes"], horizontal=True)
                    dm = st.radio("Diabetes Mellitus", ["no", "yes"], horizontal=True)
                    cad = st.radio("Coronary Artery Disease", ["no", "yes"], horizontal=True)
                with col2:
                    appet = st.radio("Appetite", ["good", "poor"], horizontal=True)
                    pe = st.radio("Pedal Edema", ["no", "yes"], horizontal=True)
                    ane = st.radio("Anemia", ["no", "yes"], horizontal=True)
                wc = st.number_input("WBC Count", 2000, 25000, 8000)
                rc = st.number_input("RBC Count", 2.0, 8.0, 5.0, step=0.1)

            submitted = st.form_submit_button("⚡ Execute Diagnostic Analysis", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if submitted or st.session_state.prediction_made:
            patient_data = {
                'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
                'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba,
                'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
                'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
                'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
            }
            
            try:
                df_patient = pd.DataFrame([patient_data])
                df_transformed = preprocessor.transform(df_patient)
                X = df_transformed[feature_cols]
                
                ckd_prob = detector.predict_proba(X)[0, 1]
                ckd_pred = detector.predict(X)[0]
                risk_score = round(ckd_prob * 100, 1)
                
                if risk_score < 30: risk_level, risk_color = "LOW", "#10B981"
                elif risk_score < 60: risk_level, risk_color = "MODERATE", "#F59E0B"
                else: risk_level, risk_color = "HIGH", "#EF4444"
                
                egfr = preprocessor.calculate_egfr(sc, age)
                stage = preprocessor.classify_ckd_stage(egfr)
                
                st.session_state.patient_data = patient_data
                st.session_state.prediction_result = {
                    'ckd_detected': bool(ckd_pred), 'risk_score': risk_score,
                    'risk_level': risk_level, 'risk_color': risk_color,
                    'stage': stage, 'egfr': egfr,
                    'confidence': round(max(ckd_prob, 1-ckd_prob) * 100, 1)
                }
                st.session_state.prediction_made = True
            except Exception as e:
                st.error(f"Prediction Error: {e}")

        if st.session_state.prediction_made:
            res = st.session_state.prediction_result
            risk_level = res.get('risk_level', 'Unknown')
            res_class = "risk-success" if risk_level == "LOW" else "risk-warning" if risk_level == "MODERATE" else "risk-danger"
            risk_color = "#10B981" if risk_level == "LOW" else "#F59E0B" if risk_level == "MODERATE" else "#EF4444"
            risk_score = res.get('risk_score', 0)
            
            st.markdown(f'<div class="glass-card report-card {res_class}">', unsafe_allow_html=True)
            r_col1, r_col2 = st.columns([1, 1.5])
            with r_col1:
                st.markdown(f"### Diagnostic Summary")
                st.markdown(f"<h1 style='color: {risk_color}; margin: 0;'>{risk_level} RISK</h1>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Level**: {res.get('confidence', 0):.1f}%")
                st.markdown(f'<div class="risk-indicator"><div class="risk-bar" style="width: {risk_score}%; background: {risk_color};"></div></div><p style="text-align: right; font-size: 0.8rem; color: #64748B;">Risk Score: {risk_score}/100</p>', unsafe_allow_html=True)
            with r_col2:
                st.markdown("### Clinical Parameters")
                m1, m2 = st.columns(2)
                m1.metric("Predicted Stage", f"Stage {res.get('stage', 'N/A')}")
                m2.metric("eGFR", f"{res.get('egfr', 'N/A')}", help="mL/min/1.73m²")
                st.markdown("---")
                st.markdown(f"**Primary Biomarker Profile**: Creatinine {st.session_state.patient_data.get('sc', 'N/A')} mg/dL | Urea {st.session_state.patient_data.get('bu', 'N/A')} mg/dL")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Action Buttons Inline
            st.markdown('### 📄 Automated Reporting')
            st.markdown("Instantly generate a verifiable PDF clinical report containing all inputs, calculations, and AI observations.")
            try:
                from utils.pdf_generator import ClinicalReportGenerator
                pdf_gen = ClinicalReportGenerator()
                pdf_path = pdf_gen.generate_report(st.session_state.patient_data, st.session_state.prediction_result, st.session_state.explanation)
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download Official Clinical Report (PDF)", f, "RenalGuard_Report.pdf", "application/pdf", use_container_width=True, type="primary")
            except Exception as e:
                st.error(f"Generate Report Error: {e}")
            
            st.markdown(f'<br>', unsafe_allow_html=True)

            # --- SHAP Explanation Integrated Inline ---
            st.markdown('### 🔍 AI Interpretability Summary')
            st.markdown("Understanding the specific biomarker contributions to the patient's risk profile.")
            try:
                from explainability.shap_explainer import SHAPExplainer
                explainer = SHAPExplainer(detector.best_model, feature_cols)
                with st.spinner("Generating clinical evidence..."):
                    df_sample = create_sample_dataset()
                    df_processed, _ = preprocessor.fit_transform(df_sample)
                    X_sample = df_processed[feature_cols]
                    explainer.fit(X_sample, sample_size=50)
                    
                    df_patient = pd.DataFrame([st.session_state.patient_data])
                    df_transformed = preprocessor.transform(df_patient)
                    X_patient = df_transformed[feature_cols]
                    
                    explanation_data = explainer.explain_prediction(X_patient)
                    st.session_state.explanation = explanation_data
                    
                    img_b64 = explainer.generate_waterfall_plot(X_patient)
                    if img_b64:
                        st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
            except Exception as e:
                st.error(f"Interpretability Module Error: {e}")
                
            st.markdown(f'<br>', unsafe_allow_html=True)

            # --- LLM Assistant Integrated Inline ---
            st.markdown('### 💬 Consult Virtual Clinical Assistant')
            if 'chat_assistant' not in st.session_state:
                st.session_state.chat_assistant = get_chat_assistant()
            
            st.session_state.chat_assistant.set_context(
                st.session_state.patient_data,
                st.session_state.prediction_result,
                st.session_state.explanation
            )
            
            chat_container = st.container()
            with chat_container:
                st.markdown('<div class="glass-card" style="padding: 1rem;">', unsafe_allow_html=True)
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])
                
                if not st.session_state.chat_history:
                    suggestions = st.session_state.chat_assistant.get_suggested_questions()
                    sub_col1, sub_col2 = st.columns(2)
                    for i, q in enumerate(suggestions[:4]):
                        with [sub_col1, sub_col2][i % 2]:
                            if st.button(q, key=f"suggest_{i}"):
                                st.session_state.chat_history.append({'role': 'user', 'content': q})
                                response = st.session_state.chat_assistant.send_message(q)
                                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                                st.rerun()

                if prompt := st.chat_input("Ask a clinical question or about your results...", key="chat_input"):
                    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
                    with st.spinner("Thinking..."):
                        response = st.session_state.chat_assistant.send_message(prompt)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🔄 Clear Data and Rerun New Patient", type="secondary", use_container_width=True):
                st.session_state.prediction_made = False
                st.session_state.chat_history = []
                st.rerun()
                
        else:
            st.markdown("""<div class="glass-card" style="text-align: center; color: #64748B; padding: 4rem 1rem;"> <div style="font-size: 4rem; opacity: 0.3;">📋</div> <p>Waiting for Clinical Data...<br>Enter patient details and click execute to generate a diagnostic profile.</p></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_about_page():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("## ℹ️ About RenalGuard AI")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🌟 Operational Mission
    Empowering healthcare providers with actionable AI technology to detect Chronic Kidney Disease at its earliest, most treatable stages.

    ### 🏆 Technical Pedigree
    Developed for **AI4Dev '26**, a national-level hackathon at **PSG College of Technology**.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card" style="height: 100%;">### 🧪 Methodology<br>- **Models**: Gradient Boosting Ensemble (XGB & LightGBM)<br>- **XAI**: Shapley Additive Explanations (SHAP)<br>- **Framework**: Streamlit Professional</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card" style="height: 100%;">### 📚 Clinical Validation<br>- **Dataset**: UCI Chronic Kidney Disease (Official)<br>- **Standards**: KDIGO Clinical Guidelines<br>- **Formulas**: Cockcroft-Gault eGFR</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card" style="border-left: 5px solid #EF4444;">### ⚠️ System Advisory<br>This software is a **Clinical Decision Support Proof of Concept**. It does not substitute independent clinical judgment by licensed practitioners.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
