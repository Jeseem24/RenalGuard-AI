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
    initial_sidebar_state="expanded"
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
        .main-header { font-size: 3rem; color: #4F46E5; text-align: center; font-weight: 800; }
        .sub-header { font-size: 1.2rem; color: #64748B; text-align: center; margin-bottom: 2rem; }
        .hero-title {
            background: linear-gradient(135deg, #1E293B 0%, #4F46E5 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
            font-size: 4rem;
            line-height: 1.0;
            margin-bottom: 1rem;
            letter-spacing: -0.04em;
        }
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

# Import modules
try:
    from preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
    from models.ckd_detector import CKDDetector, CKDStageClassifier
    from explainability.shap_explainer import SHAPExplainer
    from utils.pdf_generator import ClinicalReportGenerator
    from utils.llm_assistant import get_chat_assistant
    MODULES_LOADED = True
except ImportError as e:
    # Quietly handle google-generativeai or fpdf error, but log others
    critical_missing = ["preprocessing", "models", "explainability"]
    is_critical = any(m in str(e) for m in critical_missing)
    if is_critical:
        st.sidebar.error(f"Critical Module Error: {e}")
    else:
        st.sidebar.info(f"Extension module omitted: {e}")
    
    # Fallback definitions to avoid NameErrors
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
    """Load or train models"""
    models_path = Path(__file__).parent.parent / 'reports' / 'models'
    models_path.mkdir(parents=True, exist_ok=True)
    
    detector_path = models_path / 'ckd_detector.joblib'
    stage_path = models_path / 'stage_classifier.joblib'
    
    # Check if models exist, if not train them
    if not detector_path.exists() or not stage_path.exists():
        with st.spinner('Training models for the first time... This may take a minute.'):
            # Create sample dataset and train
            df = create_sample_dataset()
            
            # Preprocess
            preprocessor = CKDDataPreprocessor()
            df_processed, _ = preprocessor.fit_transform(df)
            
            # Prepare features
            feature_cols = [col for col in df_processed.columns if col not in ['class', 'ckd_stage']]
            X = df_processed[feature_cols]
            y = df_processed['class']
            
            # Train detector
            detector = CKDDetector()
            detector.train(X, y)
            detector.save(detector_path)
            
            # Train stage classifier
            stage_clf = CKDStageClassifier()
            stage_clf.train(X, df_processed['ckd_stage'])
            stage_clf.save(stage_path)
            
            # Save preprocessor
            preprocessor.save(models_path / 'preprocessor.joblib')
            
            return detector, stage_clf, preprocessor, feature_cols
    else:
        # Load existing models
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
    """Main application"""
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-title">
            <span style="font-size: 2rem; margin-right: 0.5rem;">🩺</span>
            RenalGuard AI
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔍 CKD Screening", "📊 AI Explanation", "💬 AI Assistant", "📈 Data Insights", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card" style="padding: 1rem; margin-top: 2rem;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; font-size: 0.85rem; color: #475569;">
            <strong style="color: #4F46E5;">AI4Dev '26</strong><br>
            National Level Hackathon<br>
            <small>PSG College of Technology</small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load models into session state
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

    # Side-load references for convenience
    detector = st.session_state.get('detector')
    preprocessor = st.session_state.get('preprocessor')
    feature_cols = st.session_state.get('feature_cols')

    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 CKD Screening":
        show_screening_page()
    elif page == "📊 AI Explanation":
        show_explanation_page()
    elif page == "💬 AI Assistant":
        show_chat_page()
    elif page == "📈 Data Insights":
        show_insights_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Redesigned Home page with Dashboard aesthetic"""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown('<h1 class="hero-title">Precision Renal Care.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Harnessing advanced AI for early Chronic Kidney Disease detection and clinical decision support.</p>', unsafe_allow_html=True)
    
    # Action Button
    if st.button("🚀 Start Diagnostic Screening", type="primary"):
        st.session_state.page = "CKD Screening"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Impact Stats Section
    st.markdown("### 📊 Global Nephrology Impact")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-label">Global Burden</div>
            <div class="metric-value">850M</div>
            <div class="metric-label">Individuals Affected</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-label">Late Diagnosis</div>
            <div class="metric-value">90%</div>
            <div class="metric-label">Are Unaware</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-label">Annual Costs</div>
            <div class="metric-value">84B+</div>
            <div class="metric-label">USD in USA alone</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-label">India Rank</div>
            <div class="metric-value">#3</div>
            <div class="metric-label">Top cause of death</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("### 🛠️ Integrated Diagnostic Suite")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown("""
        <div class="glass-card">
            <h4>🤖 AI Detection Engine</h4>
            <p>Ensemble models (XGBoost + LightGBM) trained on gold-standard clinical data for ultra-early risk identification.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <h4>📄 Clinical Reporting</h4>
            <p>Generate professional-grade PDF reports for patients and specialists with automated clinical narratives.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with f_col2:
        st.markdown("""
        <div class="glass-card">
            <h4>🔍 SHAP Interpretability</h4>
            <p>Transparent AI that explains 'why' a risk level was assigned, visualizing key contributors like Creatinine and Hemoglobin.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <h4>💬 Clinical Assistant</h4>
            <p>Conversational AI trained in the kidney health domain to guide users through their screening results.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)


def show_screening_page():
    """Main screening page with prediction form"""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("## 🔍 Patient Screening Hub")
    st.markdown("Complete the clinical profile below for a comprehensive kidney health assessment.")
    
    # Load models
    if MODULES_LOADED:
        detector, stage_clf, preprocessor, feature_cols = load_models()
    else:
        st.error("Essential modules not loaded. Please verify installation.")
        return
    
    col_input, col_result = st.columns([1.2, 1])
    
    with col_input:
        st.markdown('<div class="glass-card" style="padding: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("### 📋 Clinical Parameters")
        
        # UX Improvement: Grouping 24 inputs into logical tabs
        tab_bio, tab_urinary, tab_history = st.tabs([
            "🧪 Primary Biomarkers", 
            "🚽 Urinary Analysis", 
            "🏥 Clinical History"
        ])
        
        with st.form("ckd_form"):
            with tab_bio:
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age (Years)", 1, 120, 45)
                    bgr = st.number_input("Blood Glucose Random (mg/dL)", 50, 500, 100)
                    bu = st.number_input("Blood Urea (mg/dL)", 1, 200, 40)
                    sc = st.number_input("Serum Creatinine (mg/dL)", 0.1, 15.0, 1.0, step=0.1)
                with col2:
                    hemo = st.number_input("Hemoglobin (gms)", 3.0, 18.0, 12.0, step=0.1)
                    sod = st.number_input("Sodium (mEq/L)", 100, 170, 140)
                    pot = st.number_input("Potassium (mEq/L)", 2.0, 8.0, 4.5, step=0.1)
                    pcv = st.number_input("Packed Cell Volume (%)", 10, 60, 40)

            with tab_urinary:
                col1, col2 = st.columns(2)
                with col1:
                    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
                    al = st.select_slider("Albumin Level", options=[0, 1, 2, 3, 4, 5], value=0)
                    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
                    ba = st.selectbox("Bacteria", ["notpresent", "present"])
                with col2:
                    su = st.select_slider("Sugar Level", options=[0, 1, 2, 3, 4, 5], value=0)
                    pc = st.selectbox("Pus Cells", ["normal", "abnormal"])
                    pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
                    bp = st.number_input("Blood Pressure (mm/Hg)", 50, 250, 120)

            with tab_history:
                col1, col2 = st.columns(2)
                with col1:
                    htn = st.radio("Hypertension", ["no", "yes"], horizontal=True)
                    dm = st.radio("Diabetes Mellitus", ["no", "yes"], horizontal=True)
                    cad = st.radio("Coronary Artery Disease", ["no", "yes"], horizontal=True)
                with col2:
                    appet = st.radio("Appetite", ["good", "poor"], horizontal=True)
                    pe = st.radio("Pedal Edema", ["no", "yes"], horizontal=True)
                    ane = st.radio("Anemia", ["no", "yes"], horizontal=True)
                
                # Additional counts for model completeness
                wc = st.number_input("WBC Count", 2000, 25000, 8000)
                rc = st.number_input("RBC Count", 2.0, 8.0, 5.0, step=0.1)

            submitted = st.form_submit_button("⚡ Run Diagnostic Analysis", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_result:
        if submitted or st.session_state.prediction_made:
            # Prepare patient data
            patient_data = {
                'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
                'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba,
                'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
                'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
                'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
            }
            
            # Create dataframe
            df_patient = pd.DataFrame([patient_data])
            
            # Transform using preprocessor
            try:
                df_transformed = preprocessor.transform(df_patient)
                X = df_transformed[feature_cols]
                
                # Predict
                ckd_prob = detector.predict_proba(X)[0, 1]
                ckd_pred = detector.predict(X)[0]
                risk_score = round(ckd_prob * 100, 1)
                
                if risk_score < 30:
                    risk_level, risk_color = "LOW", "#10B981"
                elif risk_score < 60:
                    risk_level, risk_color = "MODERATE", "#F59E0B"
                else:
                    risk_level, risk_color = "HIGH", "#EF4444"
                
                egfr = preprocessor.calculate_egfr(sc, age)
                stage = preprocessor.classify_ckd_stage(egfr)
                
                st.session_state.patient_data = patient_data
                st.session_state.prediction_result = {
                    'ckd_detected': bool(ckd_pred),
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'risk_color': risk_color,
                    'stage': stage,
                    'egfr': egfr,
                    'confidence': round(max(ckd_prob, 1-ckd_prob) * 100, 1)
                }
                st.session_state.prediction_made = True
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        
        # Results Section
        if st.session_state.prediction_made:
            res = st.session_state.prediction_result
            risk_level = res.get('risk_level', 'Unknown')
            
            # Color & Class logic
            res_class = "risk-success" if risk_level == "LOW" else "risk-warning" if risk_level == "MODERATE" else "risk-danger"
            risk_color = "#10B981" if risk_level == "LOW" else "#F59E0B" if risk_level == "MODERATE" else "#EF4444"
            risk_score = res.get('risk_score', 0)
            
            st.markdown(f'<div class="glass-card report-card {res_class}">', unsafe_allow_html=True)
            r_col1, r_col2 = st.columns([1, 1.5])
            
            with r_col1:
                st.markdown(f"### Diagnostic Summary")
                st.markdown(f"<h1 style='color: {risk_color}; margin: 0;'>{risk_level} RISK</h1>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Level**: {res.get('confidence', 0):.1f}%")
                
                # Custom Risk Meter
                st.markdown(f"""
                <div class="risk-indicator">
                    <div class="risk-bar" style="width: {risk_score}%; background: {risk_color};"></div>
                </div>
                <p style='text-align: right; font-size: 0.8rem; color: #64748B;'>Risk Score: {risk_score}/100</p>
                """, unsafe_allow_html=True)
                
            with r_col2:
                st.markdown("### Clinical Parameters")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Predicted Stage", f"Stage {res.get('stage', 'N/A')}")
                with m2:
                    st.metric("eGFR", f"{res.get('egfr', 'N/A')}", help="mL/min/1.73m²")
                
                st.markdown("---")
                st.markdown(f"**Primary Biomarker Profile**: Creatinine {st.session_state.patient_data.get('sc', 'N/A')} mg/dL | Urea {st.session_state.patient_data.get('bu', 'N/A')} mg/dL")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clinical Recommendations
            st.markdown("### 📋 Clinical Recommendations")
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            recs = {
                "LOW": ["Annual kidney function screening", "Maintain blood pressure < 120/80", "Stay physically active (150 min/week)"],
                "MODERATE": ["Bi-annual nephrologist consultation", "Urine albumin-to-creatinine ratio test", "Monitor blood glucose strictly"],
                "HIGH": ["Immediate referral to a specialist", "Diagnostic imaging (kidney ultrasound)", "Review all current medications"]
            }
            
            for i, rec in enumerate(recs.get(risk_level, recs["LOW"])):
                with [rec_col1, rec_col2, rec_col3][i]:
                    st.markdown(f"""
                    <div class="metric-tile" style="height: 100%;">
                        <p style="font-size: 0.9rem; font-weight: 500;">{rec}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Action Buttons
            a_col1, a_col2, a_col3 = st.columns(3)
            with a_col1:
                if st.button("🧠 Explain Diagnostic logic", type="primary", use_container_width=True):
                    st.session_state.page = "📊 AI Explanation"
                    st.rerun()
            with a_col2:
                if st.button("📄 Generate Report PDF", use_container_width=True):
                    if MODULES_LOADED:
                        try:
                            from utils.pdf_generator import ClinicalReportGenerator
                            with st.spinner("Generating document..."):
                                pdf_gen = ClinicalReportGenerator()
                                pdf_path = pdf_gen.generate_report(
                                    st.session_state.patient_data, 
                                    st.session_state.prediction_result,
                                    st.session_state.explanation
                                )
                                with open(pdf_path, "rb") as f:
                                    st.download_button("📥 Download Official Report", f, "RenalGuard_Report.pdf", "application/pdf", use_container_width=True)
                        except ImportError:
                            st.error("PDF generation module not found. Please ensure 'utils/pdf_generator.py' exists.")
                        except Exception as e:
                            st.error(f"Error generating PDF: {e}")
                    else:
                        st.error("Report module not loaded.")
            with a_col3:
                if st.button("💬 Consult Clinical Assistant", use_container_width=True):
                    st.session_state.page = "💬 AI Assistant"
                    st.rerun()
                    
            if st.button("🔄 Reset Screening", type="secondary", use_container_width=True):
                st.session_state.prediction_made = False
                st.rerun()
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; color: #64748B; padding: 4rem 1rem;">
                <div style="font-size: 4rem; opacity: 0.3;">📋</div>
                <p>Waiting for Clinical Data...<br>Enter patient details and click analyze to generate a risk profile.</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_explanation_page():
    """Redesigned AI Interpretability Hub"""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Diagnostic Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Visualizing the clinical evidence behind your screening results using SHAP interpretability.</p>', unsafe_allow_html=True)
    
    if st.session_state.prediction_made:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns([1.2, 1])
        
        with col_exp1:
            st.markdown("### Feature Importance Breakdown")
            st.markdown("This chart shows which biomarkers most significantly influenced your risk score.")
            # Assuming SHAPExplainer and its methods are defined elsewhere
            # and explainer object is properly initialized and fitted.
            # For this change, we'll assume SHAPExplainer is available and its methods work as expected.
            # If SHAPExplainer is not defined, this will cause a NameError.
            # The original code had explainer = SHAPExplainer(detector.best_model, feature_cols)
            # and explainer.fit(X_train, sample_size=50)
            # and explanation = explainer.explain_prediction(X_patient)
            # We need to ensure these are handled or mocked for the new code to run.
            # For now, I'll assume SHAPExplainer is imported and can be instantiated.
            
            try:
                if MODULES_LOADED and st.session_state.get('models_loaded'):
                    from explainability.shap_explainer import SHAPExplainer
                    detector = st.session_state.detector
                    feature_cols = st.session_state.feature_cols
                    
                    # Instantiate with model and features
                    explainer = SHAPExplainer(detector.best_model, feature_cols)
                    
                    # Need background data for explainer
                    with st.spinner("Generating clinical evidence..."):
                        # Just a small sample from create_sample_dataset for the background if needed
                        df_sample = create_sample_dataset()
                        df_processed, _ = st.session_state.preprocessor.fit_transform(df_sample)
                        X_sample = df_processed[feature_cols]
                        
                        explainer.fit(X_sample, sample_size=50)
                        
                        # Prepare patient data
                        df_patient = pd.DataFrame([st.session_state.patient_data])
                        df_transformed = st.session_state.preprocessor.transform(df_patient)
                        X_patient = df_transformed[feature_cols]
                        
                        # Use the correct method name from shap_explainer.py
                        img_b64 = explainer.generate_waterfall_plot(X_patient)
                        if img_b64:
                            st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
                else:
                    st.warning("Clinical modules not fully active. Demonstration plot unavailable.")
            except Exception as e:
                st.error(f"Interpretability Hub Error: {e}")
                st.info("Technical Detail: Ensure background distribution data is compatible with the ensemble model.")
                # Fallback for demonstration if SHAPExplainer is not defined
                st.markdown("*(SHAP waterfall plot placeholder)*")
            
        with col_exp2:
            st.markdown("### Top Contributor Insights")
            explanation = st.session_state.explanation
            if explanation and 'top_risk_factors' in explanation:
                for factor in explanation['top_risk_factors'][:3]:
                    st.markdown(f"""
                    <div class="metric-tile" style="margin-bottom: 1rem; text-align: left;">
                        <span style="color: var(--primary); font-weight: 700;">{factor['feature'].upper()}</span>
                        <p style="font-size: 0.9rem; margin: 0;">Contribution: {factor['contribution_pct']:.1f}%</p>
                        <p style="font-size: 0.8rem; color: #64748B;">Clinical Value: {factor['feature_value']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Additional clinical insights will appear here after analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Clinical Narrative")
        st.markdown(st.session_state.explanation.get('clinical_narrative', 'Narrative generation in progress...'))
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="glass-card" style="text-align: center; padding: 4rem;">', unsafe_allow_html=True)
        st.markdown("### No Diagnostic Data Available")
        st.markdown("Please complete a screening first to view AI interpretability insights.")
        if st.button("Go to Screening"):
            st.session_state.page = "CKD Screening"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_chat_page():
    """Redesigned AI Assistant chat page"""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Virtual Nephrologist</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Conversational clinical support module. (Note: Currently running in concept-fallback mode for evaluation).</p>', unsafe_allow_html=True)
    
    # Initialize chat assistant
    if 'chat_assistant' not in st.session_state:
        st.session_state.chat_assistant = get_chat_assistant()
        
        # Set context if prediction exists
        if st.session_state.prediction_made:
            st.session_state.chat_assistant.set_context(
                st.session_state.patient_data,
                st.session_state.prediction_result,
                st.session_state.explanation
            )
    
    # Chat container
    st.markdown('<div class="glass-card" style="min-height: 400px; padding: 1.5rem;">', unsafe_allow_html=True)
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            
    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("#### How can I assist your clinical analysis today?")
        suggestions = st.session_state.chat_assistant.get_suggested_questions()
        
        sub_col1, sub_col2 = st.columns(2)
        for i, q in enumerate(suggestions[:4]):
            with [sub_col1, sub_col2][i % 2]:
                if st.button(q, key=f"suggest_{i}"):
                    st.session_state.chat_history.append({'role': 'user', 'content': q})
                    with st.spinner("Analyzing guidance..."):
                        response = st.session_state.chat_assistant.send_message(q)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input at bottom
    if prompt := st.chat_input("Ask a clinical question or about your results..."):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_assistant.send_message(prompt)
                st.markdown(response)
        
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
    
    if st.button("🗑️ Reset Consultation Thread", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.chat_assistant.reset_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def show_insights_page():
    """Data insights and model performance page"""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("## 📊 Clinical Intelligence Dashboard")
    st.markdown("Deep dive into the UCI Chronic Kidney Disease dataset and our model architectures.")
    
    tab1, tab2 = st.tabs(["🌎 Global Dataset Trends", "🤖 Model Rigor & Validation"])
    
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Population Overview")
        
        # Load sample data
        from preprocessing.data_preprocessor import create_sample_dataset
        df = create_sample_dataset()
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Total Patient Records", len(df))
        with m_col2:
            st.metric("Biomarker Dimension", len(df.columns) - 1)
        with m_col3:
            st.metric("Prevalence (CKD)", f"{(df['class'] == 'ckd').sum() / len(df):.1%}")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Diagnosis Distribution")
            class_counts = df['class'].value_counts()
            st.bar_chart(class_counts)
        with col_c2:
            st.markdown("#### Critical Missing Values")
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                st.dataframe(missing.to_frame('Missing Count'), use_container_width=True)
            else:
                st.info("The current demonstration dataset is fully clean.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Detection Engine Metrics")
        
        if MODULES_LOADED:
            detector, stage_clf, _, _ = load_models()
            
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("Engine Accuracy", f"{detector.metrics.get('accuracy', 0):.2%}")
            with p_col2:
                st.metric("Precision (PPV)", f"{detector.metrics.get('precision', 0):.2%}")
            with p_col3:
                st.metric("Recall (Sensitivity)", f"{detector.metrics.get('recall', 0):.2%}")
            with p_col4:
                st.metric("AUC-ROC Curve", f"{detector.metrics.get('roc_auc', 0):.2%}")
            
            st.markdown("---")
            
            # Confusion matrix
            c_col_1, c_col_2 = st.columns([1, 1.5])
            with c_col_1:
                st.markdown("#### Confusion Matrix")
                cm = detector.metrics.get('confusion_matrix', [[0,0],[0,0]])
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual Not CKD', 'Actual CKD'],
                    columns=['Predicted Not CKD', 'Predicted CKD']
                )
                st.table(cm_df)
            
            with c_col_2:
                st.markdown("#### Global Feature Importance")
                importance_df = detector.get_feature_importance()
                if not importance_df.empty:
                    st.bar_chart(importance_df.set_index('feature')['importance'].head(8))
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_about_page():
    """About page"""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("## ℹ️ About RenalGuard AI")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🌟 Mission
    Our mission is to empower healthcare providers with cutting-edge AI technology to detect kidney disease at its earliest, most treatable stages. We aim to reduce the global burden of CKD through accessibility and transparency.
    
    ### 🏆 Hackathon Context
    Developed for **AI4Dev '26**, a national-level hackathon at **PSG College of Technology**, focusing on transformative technologies for global development.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("""
        ### 🧪 Technology
        - **Models**: XGBoost, LightGBM, Random Forest
        - **XAI**: SHAP (Shapley Additive Explanations)
        - **Core**: Python, Scikit-Learn
        - **UI**: Streamlit with Custom CSS
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("""
        ### 📚 Medical Reference
        - **Dataset**: UCI Chronic Kidney Disease (400 samples)
        - **Standards**: KDIGO Clinical Guidelines
        - **Formulas**: Cockcroft-Gault for eGFR estimation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card" style="border-left: 5px solid #EF4444;">', unsafe_allow_html=True)
    st.markdown("""
    ### ⚠️ Medical Disclaimer
    This software is a **Proof of Concept** for screening purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding a medical condition.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
