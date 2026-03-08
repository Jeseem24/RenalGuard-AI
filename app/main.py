"""
RenalGuard AI - Main Streamlit Application
AI-Powered Early Detection & Clinical Decision Support for Chronic Kidney Disease
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# ─── PATH SETUP ──────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

src_path = str(project_root / 'src')
if src_path in sys.path:
    sys.path.remove(src_path)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RenalGuard AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ─── LOAD EXTERNAL CSS ──────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "style.css"
    try:
        if css_path.exists():
            with open(css_path, encoding='utf-8') as f:
                st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)
        else:
            _inject_fallback_css()
    except Exception as e:
        st.error(f"CSS loading failed: {e}")
        _inject_fallback_css()


def _inject_fallback_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        html, body, .stApp * { font-family: 'Inter', -apple-system, sans-serif; }
        .stApp { background: #f0f4ff; }
        h1,h2,h3,h4,h5,h6,p,span,label,input { color: #1E293B; }

        .rg-brand h1 { font-size: 1.8rem; font-weight: 800; margin: 0; }
        .rg-brand p { font-size: 0.85rem; color: #64748B; margin: 0; }
        .rg-divider { border: none; border-top: 1px solid #E2E8F0; margin: 1rem 0; }
        .rg-hero-title { font-size: 3rem; font-weight: 900; line-height: 1.1;
                         color: #1E293B; margin-bottom: 0.5rem; }
        .rg-hero-sub { font-size: 1.1rem; color: #64748B; line-height: 1.7;
                       max-width: 700px; }

        .rg-metric {
            background: white; border-radius: 16px; padding: 1.5rem;
            text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #E2E8F0;
        }
        .rg-metric-val { font-size: 2rem; font-weight: 900; color: #3B82F6; }
        .rg-metric-lbl { font-size: 0.9rem; font-weight: 700; color: #1E293B; margin-top: 0.3rem; }
        .rg-metric-sub { font-size: 0.78rem; color: #94A3B8; margin-top: 0.2rem; }

        .rg-feature {
            background: white; border-radius: 16px; padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #E2E8F0;
            min-height: 140px;
        }
        .rg-feature h4 { font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; }
        .rg-feature p { font-size: 0.85rem; color: #64748B; line-height: 1.6; }

        .glass-card {
            background: rgba(255,255,255,0.85); border-radius: 16px;
            padding: 1.5rem; border: 1px solid rgba(226,232,240,0.6);
            box-shadow: 0 4px 16px rgba(0,0,0,0.04);
            backdrop-filter: blur(10px);
        }

        .risk-gauge {
            border-radius: 16px; padding: 1.5rem; text-align: center;
            border: 2px solid #E2E8F0; margin-bottom: 1rem;
        }
        .risk-gauge.low { background: #F0FDF4; border-color: #22C55E; }
        .risk-gauge.moderate { background: #FFFBEB; border-color: #F59E0B; }
        .risk-gauge.high { background: #FEF2F2; border-color: #EF4444; }
        .risk-gauge .level { font-size: 1.4rem; font-weight: 900; }
        .risk-gauge.low .level { color: #16A34A; }
        .risk-gauge.moderate .level { color: #D97706; }
        .risk-gauge.high .level { color: #DC2626; }
        .risk-gauge .score { font-size: 0.85rem; color: #64748B; margin-top: 0.3rem; }

        .risk-bar-container {
            width: 100%%; height: 8px; background: #E2E8F0;
            border-radius: 4px; margin-top: 0.8rem; overflow: hidden;
        }
        .risk-bar-fill { height: 100%%; border-radius: 4px; transition: width 0.8s ease; }
        .risk-bar-fill.low { background: #22C55E; }
        .risk-bar-fill.moderate { background: #F59E0B; }
        .risk-bar-fill.high { background: #EF4444; }

        .empty-state {
            text-align: center; padding: 3rem 2rem;
            background: rgba(255,255,255,0.6); border-radius: 20px;
            border: 2px dashed #CBD5E1;
        }
        .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
        .empty-state h4 { color: #334155; margin-bottom: 0.5rem; }
        .empty-state p { color: #64748B; font-size: 0.9rem; line-height: 1.6; }

        .input-section-header {
            font-size: 0.85rem; font-weight: 700; color: #1E293B;
            margin: 1.2rem 0 0.5rem 0; padding-bottom: 0.3rem;
            border-bottom: 2px solid #E2E8F0;
        }

        .chat-card {
            background: white; border: 1px solid #E2E8F0; border-radius: 12px;
            padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; cursor: pointer;
            transition: all 0.2s ease; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .chat-card:hover {
            border-color: #3B82F6; box-shadow: 0 2px 8px rgba(59,130,246,0.12);
            transform: translateY(-1px);
        }
        .chat-card .chat-icon { font-size: 1.3rem; margin-right: 0.6rem; }
        .chat-card .chat-title { font-weight: 700; font-size: 0.88rem; color: #1E293B; }
        .chat-card .chat-desc { font-size: 0.78rem; color: #94A3B8; margin-top: 0.15rem; }

        .assistant-msg {
            background: linear-gradient(135deg, #F0F7FF 0%, #F8FAFC 100%);
            border: 1px solid #DBEAFE; border-radius: 16px;
            padding: 1.2rem 1.5rem; margin-top: 0.8rem;
            line-height: 1.7; color: #1E293B; font-size: 0.92rem;
        }
        .assistant-msg h4 { margin: 0 0 0.6rem 0; font-size: 1rem; font-weight: 700; }
        .assistant-msg ul { margin: 0.4rem 0; padding-left: 1.2rem; }
        .assistant-msg li { margin-bottom: 0.3rem; }
        .assistant-msg .disclaimer {
            margin-top: 1rem; padding-top: 0.8rem;
            border-top: 1px solid #E2E8F0; font-size: 0.78rem; color: #94A3B8;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)


load_css()

# ─── FEATURE MAP ─────────────────────────────────────────────────────────────
FEATURE_MAP = {
    'age': 'Age', 'bp': 'Blood Pressure', 'sg': 'Specific Gravity', 'al': 'Albumin',
    'su': 'Sugar', 'rbc': 'Red Blood Cells', 'pc': 'Pus Cell', 'pcc': 'Pus Cell Clumps',
    'ba': 'Bacteria', 'bgr': 'Blood Glucose', 'bu': 'Blood Urea', 'sc': 'Serum Creatinine',
    'sod': 'Sodium', 'pot': 'Potassium', 'hemo': 'Hemoglobin', 'pcv': 'Packed Cell Volume',
    'wc': 'WBC Count', 'rc': 'RBC Count', 'htn': 'Hypertension', 'dm': 'Diabetes',
    'cad': 'Coronary Artery Disease', 'appet': 'Appetite', 'pe': 'Pedal Edema',
    'ane': 'Anemia', 'egfr': 'eGFR',
}

# ─── SESSION STATE DEFAULTS ──────────────────────────────────────────────────
_defaults = {
    'active_page': 0,
    'prediction_made': False,
    'patient_data': {},
    'prediction_result': {},
    'explanation': {},
    'chat_history': [],
    'report_ready_path': None,
    'shap_done': False,
    'shap_img': None,
    'plain_english': '',
    'patient_name': '',
    'scroll_to_top': False,
    'form_key_counter': 0,
    'run_analysis': False,
    'active_chat': None,       # Which mock chat card is expanded
    'validation_error': None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── SCROLL TO TOP ──────────────────────────────────────────────────────────
def scroll_to_top():
    components.html(
        """
        <script>
            try { window.parent.document.querySelector('section.main').scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
            try { window.parent.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
            try {
                const c = window.parent.document.querySelectorAll(
                    '[data-testid="stAppViewContainer"], .main, .block-container'
                );
                c.forEach(function(el) { el.scrollTo({top: 0, behavior: 'smooth'}); });
            } catch(e) {}
            try {
                window.parent.document.querySelectorAll('*').forEach(function(el) {
                    if (el.scrollTop > 0) el.scrollTo({top: 0, behavior: 'smooth'});
                });
            } catch(e) {}
        </script>
        """,
        height=0,
        scrolling=False,
    )


if st.session_state.get('scroll_to_top', False):
    scroll_to_top()
    st.session_state.scroll_to_top = False


# ─── CACHED RESOURCE LOADERS ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚕️ Loading Clinical AI Engine…")
def load_models():
    import joblib
    try:
        from src.preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
    except ImportError as e:
        return None, None, None, None
    try:
        from src.models.ckd_detector import CKDDetector, CKDStageClassifier
    except ImportError as e:
        return None, None, None, None

    mp = project_root / 'reports' / 'models'
    mp.mkdir(parents=True, exist_ok=True)
    dp, sp, pp = mp / 'ckd_detector.joblib', mp / 'stage_classifier.joblib', mp / 'preprocessor.joblib'

    if not dp.exists() or not sp.exists() or not pp.exists():
        df = create_sample_dataset()
        pre = CKDDataPreprocessor(use_knn_imputer=True)
        dfp, _ = pre.fit_transform(df)
        fc = [c for c in dfp.columns if c not in ['class', 'ckd_stage']]
        X, y = dfp[fc], dfp['class']
        det = CKDDetector(); det.train(X, y); det.save(dp)
        stg = CKDStageClassifier(); stg.train(X, dfp['ckd_stage']); stg.save(sp)
        pre.save(pp)
        return det, stg, pre, fc

    md = joblib.load(dp)
    det = CKDDetector()
    det.best_model, det.best_model_name = md['model'], md['model_name']
    det.feature_names, det.metrics = md['feature_names'], md['metrics']

    sd = joblib.load(sp)
    stg = CKDStageClassifier()
    stg.model, stg.feature_names, stg.metrics = sd['model'], sd['feature_names'], sd['metrics']

    pre = CKDDataPreprocessor.load(pp)
    return det, stg, pre, det.feature_names


@st.cache_resource(show_spinner=False)
def get_shap_background_data():
    from src.preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
    ds = create_sample_dataset()
    bg_pre = CKDDataPreprocessor(use_knn_imputer=True)
    dsp, _ = bg_pre.fit_transform(ds)
    fc = [c for c in dsp.columns if c not in ['class', 'ckd_stage']]
    return dsp[fc]


def _key(name: str) -> str:
    return f"{name}_{st.session_state.form_key_counter}"


# ═════════════════════════════════════════════════════════════════════════════
# SMART MOCK CHAT ASSISTANT — generates contextual responses from results
# ═════════════════════════════════════════════════════════════════════════════

def _generate_mock_responses(patient_data: dict, result: dict, explanation: dict) -> dict:
    """
    Build 4 contextual chat responses based on actual patient results.
    Each response is dynamically generated — not static boilerplate.
    """
    risk   = result.get('risk_score', 0)
    stage  = result.get('stage', 0)
    egfr   = result.get('egfr', 90)
    level  = result.get('risk_level', 'LOW')
    name   = patient_data.get('patient_name', 'the patient')

    sc   = patient_data.get('sc', 1.0)
    hemo = patient_data.get('hemo', 14.0)
    bp   = patient_data.get('bp', 80)
    bgr  = patient_data.get('bgr', 100)
    bu   = patient_data.get('bu', 30)
    al   = patient_data.get('al', 0)
    sod  = patient_data.get('sod', 137)
    pot  = patient_data.get('pot', 4.0)
    htn  = patient_data.get('htn', 'no')
    dm   = patient_data.get('dm', 'no')
    ane  = patient_data.get('ane', 'no')
    pe   = patient_data.get('pe', 'no')

    # ── Collect abnormal findings ─────────────
    abnormals = []
    if sc > 1.4:    abnormals.append(f"elevated serum creatinine ({sc} mg/dL)")
    if hemo < 12.0: abnormals.append(f"low hemoglobin ({hemo} g/dL)")
    if bp > 140:    abnormals.append(f"elevated blood pressure ({bp} mmHg)")
    if bgr > 200:   abnormals.append(f"high blood glucose ({bgr} mg/dL)")
    if bu > 50:     abnormals.append(f"elevated blood urea ({bu} mg/dL)")
    if al > 2:      abnormals.append(f"significant albuminuria (level {al})")
    if pot > 5.5:   abnormals.append(f"hyperkalemia ({pot} mEq/L)")
    if sod < 125:   abnormals.append(f"hyponatremia ({sod} mEq/L)")

    abnormal_text = ", ".join(abnormals) if abnormals else "no critical abnormalities"

    # ── STAGE DESCRIPTIONS ────────────────────
    stage_info = {
        0: ("Normal / No CKD", "≥90 mL/min with no kidney damage markers"),
        1: ("Stage 1 — Kidney damage with normal GFR", "≥90 mL/min but with albuminuria or structural damage"),
        2: ("Stage 2 — Mild reduction", "60–89 mL/min; monitor annually"),
        3: ("Stage 3 — Moderate reduction", "30–59 mL/min; nephrology referral recommended"),
        4: ("Stage 4 — Severe reduction", "15–29 mL/min; prepare for renal replacement therapy"),
        5: ("Stage 5 — Kidney failure", "<15 mL/min; dialysis or transplant evaluation needed"),
    }
    stage_name, stage_desc = stage_info.get(stage, ("Unknown", ""))

    # ═══════════════════════════════════════════
    # RESPONSE 1: Results Summary
    # ═══════════════════════════════════════════
    if level == "LOW":
        summary_body = f"""
<h4>📊 Results Summary for {name}</h4>
<p>The AI screening indicates a <strong style="color:#16A34A;">LOW risk</strong> of Chronic Kidney Disease
with a risk score of <strong>{risk}%</strong> and model confidence of <strong>{result.get('confidence',0)}%</strong>.</p>
<ul>
    <li><strong>eGFR:</strong> {egfr} mL/min — within normal range</li>
    <li><strong>CKD Stage:</strong> {stage_name}</li>
    <li><strong>Serum Creatinine:</strong> {sc} mg/dL</li>
    <li><strong>Key Findings:</strong> {abnormal_text}</li>
</ul>
<p>✅ <strong>Interpretation:</strong> Current biomarkers suggest healthy kidney function. Continue routine
monitoring as part of standard preventive care. Annual screening is recommended if the patient has
diabetes, hypertension, or family history of CKD.</p>
<div class="disclaimer">⚠️ This is an AI screening tool, not a clinical diagnosis. Always confirm with laboratory testing.</div>
"""
    elif level == "MODERATE":
        summary_body = f"""
<h4>📊 Results Summary for {name}</h4>
<p>The AI screening indicates a <strong style="color:#D97706;">MODERATE risk</strong> of Chronic Kidney Disease
with a risk score of <strong>{risk}%</strong> and model confidence of <strong>{result.get('confidence',0)}%</strong>.</p>
<ul>
    <li><strong>eGFR:</strong> {egfr} mL/min — {"mildly reduced" if egfr < 90 else "borderline"}</li>
    <li><strong>CKD Stage:</strong> {stage_name}</li>
    <li><strong>Abnormal Findings:</strong> {abnormal_text}</li>
</ul>
<p>⚠️ <strong>Interpretation:</strong> Several biomarkers are outside optimal ranges. This does not confirm CKD
but warrants further investigation. Recommend repeat testing in 2–4 weeks and consider nephrology consultation
if abnormalities persist.</p>
<div class="disclaimer">⚠️ This is an AI screening tool, not a clinical diagnosis. Always confirm with laboratory testing.</div>
"""
    else:
        summary_body = f"""
<h4>📊 Results Summary for {name}</h4>
<p>The AI screening indicates a <strong style="color:#DC2626;">HIGH risk</strong> of Chronic Kidney Disease
with a risk score of <strong>{risk}%</strong> and model confidence of <strong>{result.get('confidence',0)}%</strong>.</p>
<ul>
    <li><strong>eGFR:</strong> {egfr} mL/min — {"severely reduced" if egfr < 30 else "significantly reduced"}</li>
    <li><strong>CKD Stage:</strong> {stage_name}</li>
    <li><strong>Critical Findings:</strong> {abnormal_text}</li>
</ul>
<p>🔴 <strong>Interpretation:</strong> Multiple biomarkers indicate significant kidney function impairment.
<strong>Urgent nephrology referral is recommended.</strong> Further testing should include urine albumin-to-creatinine
ratio (UACR), renal ultrasound, and comprehensive metabolic panel.</p>
<div class="disclaimer">⚠️ This is an AI screening tool, not a clinical diagnosis. Urgent clinical evaluation recommended.</div>
"""

    # ═══════════════════════════════════════════
    # RESPONSE 2: Dietary & Lifestyle Guidance
    # ═══════════════════════════════════════════
    if stage <= 1:
        diet_body = f"""
<h4>🥗 Dietary & Lifestyle Recommendations</h4>
<p>Based on {name}'s <strong>Stage {stage}</strong> (eGFR: {egfr} mL/min), here are evidence-based recommendations:</p>
<ul>
    <li>🥤 <strong>Hydration:</strong> Maintain adequate water intake (2–2.5L/day unless contraindicated)</li>
    <li>🧂 <strong>Sodium:</strong> Limit to &lt;2,300 mg/day to support blood pressure control</li>
    <li>🥩 <strong>Protein:</strong> Normal protein intake is appropriate at this stage (0.8–1.0 g/kg/day)</li>
    <li>🍎 <strong>Diet Pattern:</strong> DASH or Mediterranean diet — rich in fruits, vegetables, whole grains</li>
    <li>🏃 <strong>Exercise:</strong> 150 minutes/week of moderate aerobic activity</li>
    {"<li>📉 <strong>Blood Pressure:</strong> Target &lt;130/80 mmHg given current reading of " + str(bp) + " mmHg</li>" if bp > 120 else ""}
    {"<li>💉 <strong>Glucose Control:</strong> Optimize glycemic management (current: " + str(bgr) + " mg/dL)</li>" if dm == 'yes' or bgr > 140 else ""}
</ul>
<p>💡 <strong>Key message:</strong> At this stage, healthy lifestyle choices are the most powerful tool for
preventing progression.</p>
<div class="disclaimer">⚠️ Dietary advice should be personalized by a registered dietitian based on complete clinical assessment.</div>
"""
    elif stage <= 3:
        diet_body = f"""
<h4>🥗 Dietary & Lifestyle Recommendations</h4>
<p>Based on {name}'s <strong>Stage {stage}</strong> (eGFR: {egfr} mL/min), modified dietary management is recommended:</p>
<ul>
    <li>🥩 <strong>Protein:</strong> Moderate restriction to 0.6–0.8 g/kg/day to reduce kidney workload</li>
    <li>🧂 <strong>Sodium:</strong> Strict limit to &lt;2,000 mg/day</li>
    <li>🍌 <strong>Potassium:</strong> {"⚠️ RESTRICT — current level elevated at " + str(pot) + " mEq/L. Avoid bananas, oranges, potatoes, tomatoes" if pot > 5.0 else "Monitor levels; moderate intake of high-potassium foods"}</li>
    <li>🥛 <strong>Phosphorus:</strong> Limit dairy, processed foods, dark colas</li>
    <li>🥤 <strong>Fluids:</strong> May need adjustment — consult nephrologist</li>
    <li>🏃 <strong>Exercise:</strong> Continue moderate activity as tolerated</li>
    {"<li>📉 <strong>Blood Pressure:</strong> Target &lt;130/80 mmHg (current: " + str(bp) + " mmHg). ACEi/ARB therapy may be beneficial</li>" if htn == 'yes' or bp > 130 else ""}
    {"<li>🩸 <strong>Anemia:</strong> Monitor iron studies; ESA therapy may be needed (Hgb: " + str(hemo) + " g/dL)</li>" if hemo < 11.0 else ""}
</ul>
<div class="disclaimer">⚠️ Dietary advice should be personalized by a registered dietitian and nephrologist.</div>
"""
    else:
        diet_body = f"""
<h4>🥗 Dietary & Lifestyle Recommendations</h4>
<p>Based on {name}'s <strong>Stage {stage}</strong> (eGFR: {egfr} mL/min), strict dietary management is critical:</p>
<ul>
    <li>🥩 <strong>Protein:</strong> Strict restriction to 0.6 g/kg/day (or per dialysis protocol if applicable)</li>
    <li>🧂 <strong>Sodium:</strong> &lt;1,500 mg/day — avoid processed foods entirely</li>
    <li>🍌 <strong>Potassium:</strong> Strict restriction — current: {pot} mEq/L. Avoid high-K foods</li>
    <li>🥛 <strong>Phosphorus:</strong> &lt;800 mg/day. Phosphate binders may be needed</li>
    <li>🥤 <strong>Fluids:</strong> Likely restricted — follow nephrologist guidance</li>
    <li>{"🩸 <strong>Anemia Management:</strong> EPO/ESA therapy likely needed (Hgb: " + str(hemo) + " g/dL)" if hemo < 10.0 else "🩸 <strong>Anemia:</strong> Monitor closely"}</li>
    {"<li>🦵 <strong>Edema:</strong> Elevate legs; compression stockings; diuretic therapy as prescribed</li>" if pe == 'yes' else ""}
</ul>
<p>🔴 <strong>Important:</strong> At Stage {stage}, close coordination between nephrologist, dietitian, and
primary care is essential. Renal replacement therapy planning should be discussed.</p>
<div class="disclaimer">⚠️ Advanced CKD requires specialist-managed dietary protocols. This is general guidance only.</div>
"""

    # ═══════════════════════════════════════════
    # RESPONSE 3: Recommended Next Steps
    # ═══════════════════════════════════════════
    next_steps = []
    if stage >= 3:
        next_steps.append("🏥 <strong>Urgent nephrology referral</strong> for comprehensive kidney evaluation")
    elif stage >= 2 or risk >= 40:
        next_steps.append("🏥 <strong>Nephrology consultation</strong> recommended within 2–4 weeks")
    else:
        next_steps.append("📋 <strong>Routine follow-up</strong> with primary care in 6–12 months")

    next_steps.append(f"🔬 <strong>Repeat labs in {'2 weeks' if risk >= 60 else '3 months'}:</strong> "
                      "BMP, CBC, urinalysis with microscopy")
    next_steps.append("📊 <strong>UACR test:</strong> Urine albumin-to-creatinine ratio to quantify proteinuria")

    if sc > 1.4 or egfr < 60:
        next_steps.append("🩻 <strong>Renal ultrasound:</strong> Assess kidney size, structure, and rule out obstruction")

    if htn == 'yes' or bp > 140:
        next_steps.append(f"💊 <strong>Blood pressure optimization:</strong> Current {bp} mmHg → target &lt;130/80. "
                         "Consider ACEi/ARB initiation or dose adjustment")

    if dm == 'yes' or bgr > 200:
        next_steps.append(f"💉 <strong>Glycemic control:</strong> Current glucose {bgr} mg/dL. "
                         "HbA1c target &lt;7% to slow CKD progression")

    if ane == 'yes' or hemo < 11.0:
        next_steps.append(f"🩸 <strong>Anemia workup:</strong> Iron studies, B12, folate (Hgb: {hemo} g/dL)")

    if stage >= 4:
        next_steps.append("🫘 <strong>Renal replacement planning:</strong> Discuss dialysis access, transplant evaluation")

    next_steps_html = "\n".join([f"<li>{s}</li>" for s in next_steps])
    next_body = f"""
<h4>🔮 Recommended Next Steps for {name}</h4>
<p>Based on a <strong>{level} risk</strong> assessment (Stage {stage}, eGFR {egfr} mL/min):</p>
<ol>
{next_steps_html}
</ol>
<p>📅 <strong>Follow-up timeline:</strong> {"Within 1–2 weeks" if risk >= 60 else "Within 1–3 months" if risk >= 30 else "Routine annual screening"}</p>
<div class="disclaimer">⚠️ Clinical decisions should be made by the treating physician based on complete patient evaluation.</div>
"""

    # ═══════════════════════════════════════════
    # RESPONSE 4: Understanding Key Biomarkers
    # ═══════════════════════════════════════════
    biomarker_rows = []

    # Always show these core biomarkers
    cr_status = "🟢 Normal" if 0.6 <= sc <= 1.4 else "🔴 Elevated" if sc > 1.4 else "🟡 Low"
    biomarker_rows.append(f"<li><strong>Serum Creatinine ({sc} mg/dL):</strong> {cr_status} — "
                         f"Waste product filtered by kidneys. Elevated levels suggest reduced filtration capacity. "
                         f"Normal range: 0.6–1.4 mg/dL.</li>")

    egfr_status = "🟢 Normal" if egfr >= 90 else "🟡 Mildly reduced" if egfr >= 60 else "🔴 Significantly reduced"
    biomarker_rows.append(f"<li><strong>eGFR ({egfr} mL/min):</strong> {egfr_status} — "
                         f"Estimated Glomerular Filtration Rate measures how well kidneys filter blood. "
                         f"Normal: ≥90 mL/min.</li>")

    hb_status = "🟢 Normal" if hemo >= 12 else "🔴 Low (anemic)"
    biomarker_rows.append(f"<li><strong>Hemoglobin ({hemo} g/dL):</strong> {hb_status} — "
                         f"Kidneys produce erythropoietin; CKD often causes anemia. Normal: 12–17 g/dL.</li>")

    bu_status = "🟢 Normal" if bu <= 50 else "🔴 Elevated"
    biomarker_rows.append(f"<li><strong>Blood Urea ({bu} mg/dL):</strong> {bu_status} — "
                         f"Protein metabolism waste product. Rises when kidney filtration declines. Normal: 7–20 mg/dL.</li>")

    if al > 0:
        al_status = "🟡 Mild" if al <= 2 else "🔴 Significant"
        biomarker_rows.append(f"<li><strong>Albumin in urine (Level {al}):</strong> {al_status} — "
                             f"Protein leak in urine is an early marker of kidney damage.</li>")

    if pot > 5.0 or pot < 3.5:
        k_status = "🔴 High" if pot > 5.0 else "🟡 Low"
        biomarker_rows.append(f"<li><strong>Potassium ({pot} mEq/L):</strong> {k_status} — "
                             f"Kidneys regulate potassium. Abnormal levels can cause cardiac arrhythmias. "
                             f"Normal: 3.5–5.0 mEq/L.</li>")

    biomarker_html = "\n".join(biomarker_rows)
    biomarker_body = f"""
<h4>🧬 Understanding {name}'s Key Biomarkers</h4>
<p>Here's what each important biomarker means in this patient's context:</p>
<ul>
{biomarker_html}
</ul>
<p>💡 <strong>How to read the results:</strong> 🟢 = within normal range, 🟡 = borderline/mild concern,
🔴 = requires attention. Multiple red indicators together increase clinical significance.</p>
<div class="disclaimer">⚠️ Biomarker interpretation should always consider the full clinical picture, patient history, and trends over time.</div>
"""

    return {
        'summary':    summary_body,
        'diet':       diet_body,
        'next_steps': next_body,
        'biomarkers': biomarker_body,
    }


# ─── NAVIGATION ───────────────────────────────────────────────────────────────
NAV = ["🏠 Home", "🩺 Screening", "ℹ️ About"]


def render_nav():
    st.markdown("""
    <div class="rg-brand">
        <h1>🩺 RenalGuard AI</h1>
        <p>Clinical Decision Support System • v2.0</p>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(len(NAV))
    for i, (col, label) in enumerate(zip(cols, NAV)):
        btn_type = "primary" if st.session_state.active_page == i else "secondary"
        if col.button(label, type=btn_type, use_container_width=True, key=f"nav_{i}"):
            if st.session_state.active_page != i:
                st.session_state.active_page = i
                st.session_state.scroll_to_top = True
                st.rerun()
    st.markdown("<hr class='rg-divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown(
        '<p class="rg-hero-title">Precision Kidney<br>Health Screening.</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="rg-hero-sub">AI-powered early detection of Chronic Kidney Disease '
        'using 24 clinical biomarkers. Instant risk stratification, transparent explanations, '
        'and professional reporting — designed for primary care.</p>',
        unsafe_allow_html=True
    )

    c1, c2, _ = st.columns([1.2, 1.2, 2])
    if c1.button("🚀 Start Screening", type="primary", use_container_width=True, key="hero_start"):
        st.session_state.active_page = 1
        st.session_state.scroll_to_top = True
        st.rerun()
    if c2.button("ℹ️ Learn More", type="secondary", use_container_width=True, key="hero_learn"):
        st.session_state.active_page = 2
        st.session_state.scroll_to_top = True
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 The Global CKD Crisis")

    tiles = [
        ("850M+", "People Affected", "Worldwide prevalence"),
        ("90%",   "Diagnosed Late",  "When damage is irreversible"),
        ("$84B+", "Annual Cost",     "In the USA alone"),
        ("10×",   "Cost Reduction",  "With early detection"),
    ]
    cols = st.columns(4)
    for col, (val, lbl, sub) in zip(cols, tiles):
        col.markdown(
            f'<div class="rg-metric">'
            f'<div class="rg-metric-val">{val}</div>'
            f'<div class="rg-metric-lbl">{lbl}</div>'
            f'<div class="rg-metric-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### 🛠️ Platform Capabilities")
    features = [
        ("🤖", "AI Ensemble Detection",
         "XGBoost + LightGBM stacking ensemble with 5-fold cross-validation for robust CKD risk identification."),
        ("🔍", "Plain-English Insights",
         "SHAP explainability translated into language any patient or GP can understand — no black boxes."),
        ("📄", "1-Click PDF Reports",
         "Professional clinical screening reports with lab values, risk factors, and KDIGO-aligned recommendations."),
        ("💬", "Smart Assistant",
         "Context-aware clinical assistant providing stage-specific dietary guidance and next-step recommendations."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f'<div class="rg-feature"><h4>{icon} {title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SCREENING
# ═════════════════════════════════════════════════════════════════════════════
def page_screening():
    detector, stage_clf, preprocessor, feature_cols = load_models()

    if st.session_state.get('validation_error'):
        st.error(st.session_state.validation_error)
        st.session_state.validation_error = None

    if detector is None:
        st.error("⚠️ AI models could not be loaded. Run `python scripts/train_models.py` first.")
        return

    st.markdown("### 🩺 Clinical Screening Workstation")
    st.caption("Enter patient biomarkers → receive instant AI-powered risk assessment with explanations.")
    st.markdown("---")

    left, right = st.columns([1, 1.15], gap="large")

    # ═══════════════════════════════════════════
    # LEFT: INPUT PANEL
    # ═══════════════════════════════════════════
    with left:
        st.markdown("##### 📋 Patient Biomarker Input")

        st.markdown('<div class="input-section-header">👤 Patient Information</div>',
                    unsafe_allow_html=True)
        patient_name = st.text_input(
            "Patient Full Name",
            value=st.session_state.get('patient_name', ''),
            placeholder="e.g., John Doe",
            help="Enter the patient's full name for the clinical report",
            key=_key("patient_name"),
        )

        st.markdown('<div class="input-section-header">📊 Demographics & Vitals</div>',
                    unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        age = d1.number_input("Age (years)",           2,   120, 48,  key=_key("age"))
        bp  = d2.number_input("Blood Pressure (mmHg)", 40,  250, 80,  key=_key("bp"))

        st.markdown('<div class="input-section-header">🩸 Blood Chemistry</div>',
                    unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        bgr  = b1.number_input("Blood Glucose (mg/dL)",       20,    500,   117,          key=_key("bgr"))
        bu   = b2.number_input("Blood Urea (mg/dL)",          1.0,   400.0, 31.0,  step=0.5, key=_key("bu"))
        sc   = b1.number_input("Serum Creatinine (mg/dL)",    0.1,   50.0,  1.1,   step=0.1, key=_key("sc"))
        sod  = b2.number_input("Sodium (mEq/L)",              100,   180,   137,          key=_key("sod"))
        pot  = b1.number_input("Potassium (mEq/L)",           2.0,   10.0,  4.0,   step=0.1, key=_key("pot"))
        hemo = b2.number_input("Hemoglobin (g/dL)",           3.0,   20.0,  14.1,  step=0.1, key=_key("hemo"))
        pcv  = b1.number_input("Packed Cell Volume (%)",      10,    65,    43,           key=_key("pcv"))
        wc   = b2.number_input("WBC Count (cells/cumm)",      2000,  30000, 8000,  step=100, key=_key("wc"))
        rc   = b1.number_input("RBC Count (millions/cmm)",    2.0,   8.0,   5.0,   step=0.1, key=_key("rc"))

        st.markdown('<div class="input-section-header">🔬 Urinalysis</div>',
                    unsafe_allow_html=True)
        u1, u2 = st.columns(2)
        sg = u1.selectbox("Specific Gravity",
                          [1.005, 1.010, 1.015, 1.020, 1.025], index=3, key=_key("sg"))
        al = u2.selectbox("Albumin Level",
                          [0, 1, 2, 3, 4, 5], index=0, key=_key("al"))
        su = u1.selectbox("Sugar Level",
                          [0, 1, 2, 3, 4, 5], index=0, key=_key("su"))

        st.markdown('<div class="input-section-header">🏥 Clinical History & Examination</div>',
                    unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        rbc_v = ch1.radio("RBC in Urine",            ["normal", "abnormal"],    horizontal=True, key=_key("rbc"))
        pc_v  = ch2.radio("Pus Cells",               ["normal", "abnormal"],    horizontal=True, key=_key("pc"))
        pcc   = ch1.radio("Pus Cell Clumps",          ["notpresent", "present"], horizontal=True, key=_key("pcc"))
        ba    = ch2.radio("Bacteria",                 ["notpresent", "present"], horizontal=True, key=_key("ba"))
        htn   = ch1.radio("Hypertension",             ["no", "yes"],             horizontal=True, key=_key("htn"))
        dm    = ch2.radio("Diabetes Mellitus",        ["no", "yes"],             horizontal=True, key=_key("dm"))
        cad   = ch1.radio("Coronary Artery Disease",  ["no", "yes"],             horizontal=True, key=_key("cad"))
        appet = ch2.radio("Appetite",                 ["good", "poor"],          horizontal=True, key=_key("appet"))
        pe    = ch1.radio("Pedal Edema",              ["no", "yes"],             horizontal=True, key=_key("pe"))
        ane   = ch2.radio("Anemia",                   ["no", "yes"],             horizontal=True, key=_key("ane"))

        st.markdown("")

        run_clicked = st.button(
            "⚡ Run Diagnostic Analysis",
            type="primary",
            use_container_width=True,
            key=_key("run_btn"),
        )

    # ═══════════════════════════════════════════
    # RIGHT: RESULTS PANEL
    # ═══════════════════════════════════════════
    with right:

        if run_clicked:
            clean_name = patient_name.strip() if patient_name else ""
            if not clean_name:
                st.session_state.validation_error = "❌ Please enter the patient's full name before running the analysis."
                st.session_state.scroll_to_top = True
                st.rerun()

            st.session_state.patient_name = clean_name

            for k in ['shap_done', 'shap_img', 'explanation', 'report_ready_path',
                       'chat_history', 'plain_english', 'active_chat']:
                st.session_state[k] = _defaults[k]

            patient_data = {
                'patient_name': clean_name,
                'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
                'rbc': rbc_v, 'pc': pc_v, 'pcc': pcc, 'ba': ba,
                'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
                'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
                'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet,
                'pe': pe, 'ane': ane,
            }

            validation = preprocessor.validate_input(patient_data)
            if validation['errors']:
                st.session_state.validation_error = f"❌ {validation['errors'][0]}"
                st.session_state.scroll_to_top = True
                st.rerun()
            if validation['warnings']:
                for warn in validation['warnings']:
                    st.warning(f"⚠️ {warn}")

            with st.spinner("🔬 Running AI Diagnostic Analysis…"):
                try:
                    df_pt = pd.DataFrame(
                        [{k: v for k, v in patient_data.items() if k != 'patient_name'}]
                    )
                    df_tx = preprocessor.transform(df_pt)
                    for col in feature_cols:
                        if col not in df_tx.columns:
                            df_tx[col] = 0
                    X = df_tx[feature_cols]

                    ckd_prob = detector.predict_proba(X)[0, 1]
                    risk = round(ckd_prob * 100, 1)

                    if   risk < 30: lvl, css_class = "LOW",      "low"
                    elif risk < 60: lvl, css_class = "MODERATE",  "moderate"
                    else:           lvl, css_class = "HIGH",      "high"

                    egfr  = preprocessor.calculate_egfr(sc, age)
                    stage = preprocessor.classify_ckd_stage(egfr)

                    if risk >= 30 and stage_clf is not None:
                        try:
                            ml_stage = int(stage_clf.predict(X)[0])
                            stage = max(stage, ml_stage)
                        except Exception:
                            pass

                    st.session_state.patient_data = patient_data
                    st.session_state.prediction_result = {
                        'ckd_detected': risk >= 50,
                        'risk_score':   risk,
                        'risk_level':   lvl,
                        'css_class':    css_class,
                        'stage':        stage,
                        'egfr':         egfr,
                        'confidence':   round(max(ckd_prob, 1 - ckd_prob) * 100, 1),
                    }
                    st.session_state.prediction_made = True
                    st.session_state.scroll_to_top = True
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Prediction Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # ── Empty state ───────────────────────
        if not st.session_state.prediction_made:
            st.markdown(
                '<div class="empty-state">'
                '<div class="icon">🔬</div>'
                '<h4>Awaiting Patient Data</h4>'
                '<p>Fill in the clinical biomarkers on the left panel, then click '
                '<strong>Run Diagnostic Analysis</strong> to receive your '
                'AI-powered assessment.</p>'
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown("")
            st.markdown("**What you'll receive:**")
            st.markdown(
                "- 🎯 CKD risk level (Low / Moderate / High)\n"
                "- 📊 eGFR calculation & CKD staging (CKD-EPI 2021)\n"
                "- 🔍 Plain-English explanation of key risk factors\n"
                "- 📄 Downloadable PDF clinical report\n"
                "- 💬 AI assistant for follow-up questions"
            )
            return

        # ══════════════════════════════════════
        # DISPLAY RESULTS
        # ══════════════════════════════════════
        res = st.session_state.prediction_result
        pt  = st.session_state.patient_data

        # ── Patient Banner ────────────────────
        p_name = pt.get('patient_name', 'Unknown Patient')
        st.markdown(
            f'<div class="glass-card" style="padding: 1rem 1.5rem; margin-bottom: 1rem;">'
            f'<span style="font-size: 0.75rem; font-weight: 600; '
            f'text-transform: uppercase; letter-spacing: 0.08em; color: #64748B;">'
            f'Patient</span><br>'
            f'<span style="font-size: 1.3rem; font-weight: 800; color: #1E293B;">'
            f'{p_name}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── 1. Risk Gauge ─────────────────────
        st.markdown("##### 🎯 Risk Assessment")
        risk_score = res['risk_score']
        st.markdown(
            f'<div class="risk-gauge {res["css_class"]}">'
            f'<div class="level">{res["risk_level"]} RISK</div>'
            f'<div class="score">Score: {risk_score:.1f}/100  •  '
            f'Confidence: {res["confidence"]:.1f}%</div>'
            f'<div class="risk-bar-container">'
            f'<div class="risk-bar-fill {res["css_class"]}" '
            f'style="width: {risk_score}%;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("CKD Stage", f"Stage {res['stage']}",
                  help=preprocessor.get_stage_description(res['stage']))
        m2.metric("eGFR", f"{res['egfr']} mL/min",
                  help="Estimated Glomerular Filtration Rate (CKD-EPI 2021)")
        sc_status = preprocessor.get_value_status('sc', pt.get('sc', 0))
        m3.metric("Creatinine", f"{pt.get('sc', '?')} mg/dL",
                  delta="Normal" if sc_status == 'NORMAL' else "Abnormal",
                  delta_color="normal" if sc_status == 'NORMAL' else "inverse")

        st.markdown("---")

        # ── 2. SHAP Explanation ───────────────
        st.markdown("##### 🔍 AI Explanation")
        if not st.session_state.shap_done:
            with st.spinner("Analyzing biomarker contributions…"):
                try:
                    from src.explainability.shap_explainer import SHAPExplainer
                    exp = SHAPExplainer(detector.best_model, feature_cols)
                    bg_data = get_shap_background_data()
                    for c in feature_cols:
                        if c not in bg_data.columns:
                            bg_data[c] = 0
                    bg_data = bg_data[feature_cols]
                    exp.fit(bg_data, sample_size=80)

                    df_pt = pd.DataFrame(
                        [{k: v for k, v in pt.items() if k != 'patient_name'}]
                    )
                    df_tx = preprocessor.transform(df_pt)
                    for col in feature_cols:
                        if col not in df_tx.columns:
                            df_tx[col] = 0
                    X_pt = df_tx[feature_cols]

                    exp_data = exp.explain_prediction(X_pt)
                    plain    = exp.generate_plain_english(exp_data, res['risk_level'])
                    b64      = exp.generate_waterfall_plot(X_pt)

                    st.session_state.explanation   = exp_data
                    st.session_state.plain_english = plain
                    st.session_state.shap_img      = b64
                    st.session_state.shap_done     = True
                except Exception as e:
                    st.session_state.shap_done = True
                    st.caption(f"⚠️ SHAP explanation unavailable: {e}")

        # ── Render SHAP plain-English text ────
        if st.session_state.plain_english:
            st.markdown(st.session_state.plain_english)

        # ── Render SHAP waterfall plot image ──
        if st.session_state.shap_img:
            try:
                import base64
                img_b64 = st.session_state.shap_img
                # Handle both raw base64 string and data-URI prefix
                if img_b64.startswith('data:'):
                    img_src = img_b64
                else:
                    img_src = f"data:image/png;base64,{img_b64}"

                st.markdown(
                    f'<div class="glass-card" style="padding: 1rem; margin-top: 0.8rem;">'
                    f'<p style="font-size: 0.8rem; font-weight: 700; color: #64748B; '
                    f'margin-bottom: 0.6rem;">📊 SHAP Feature Importance — Waterfall Plot</p>'
                    f'<img src="{img_src}" style="width: 100%; border-radius: 8px;" '
                    f'alt="SHAP Waterfall Plot"/>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.caption(f"⚠️ Could not render SHAP plot: {e}")
        elif st.session_state.shap_done and not st.session_state.plain_english:
            st.info("ℹ️ Detailed biomarker explanation is not available for this prediction.")

        st.markdown("---")

        # ── 3. PDF Report ─────────────────────
        st.markdown("##### 📄 Clinical Report")
        rp1, rp2 = st.columns(2)

        if rp1.button("📝 Generate PDF", type="primary", use_container_width=True, key="gen_pdf"):
            try:
                from src.utils.pdf_generator import ClinicalReportGenerator
                gen = ClinicalReportGenerator()
                pdf_path = gen.generate_report(pt, res, st.session_state.explanation)
                st.session_state.report_ready_path = pdf_path
                st.rerun()
            except Exception as e:
                st.error(f"Report generation error: {e}")

        if st.session_state.report_ready_path:
            try:
                with open(st.session_state.report_ready_path, "rb") as f:
                    rp2.download_button(
                        "📥 Download Report", f,
                        file_name="RenalGuard_Report.pdf",
                        mime="application/pdf",
                        key="dl_pdf",
                    )
            except FileNotFoundError:
                rp2.warning("Report file not found. Please regenerate.")
                st.session_state.report_ready_path = None

        st.markdown("---")

        # ── 4. Smart Clinical Assistant (Mock) ─
        st.markdown("##### 💬 Clinical Assistant")
        st.caption("Select a topic to get AI-generated insights based on this patient's results.")

        # Generate contextual responses
        mock_responses = _generate_mock_responses(pt, res, st.session_state.explanation)

        # Chat topic cards
        chat_topics = [
            ("summary",    "📊", "Results Summary",        "Complete overview of the screening findings"),
            ("diet",       "🥗", "Diet & Lifestyle",       "Stage-specific dietary and lifestyle recommendations"),
            ("next_steps", "🔮", "Recommended Next Steps",  "Follow-up tests, referrals, and timeline"),
            ("biomarkers", "🧬", "Biomarker Explainer",     "What each key lab value means for this patient"),
        ]

        # Render 4 clickable cards
        card_cols = st.columns(2)
        for idx, (topic_key, icon, title, desc) in enumerate(chat_topics):
            col = card_cols[idx % 2]
            with col:
                btn_key = f"chat_{topic_key}_{st.session_state.form_key_counter}"
                if st.button(
                    f"{icon} {title}",
                    key=btn_key,
                    use_container_width=True,
                    type="secondary" if st.session_state.active_chat != topic_key else "primary",
                ):
                    st.session_state.active_chat = (
                        None if st.session_state.active_chat == topic_key else topic_key
                    )
                    st.rerun()
                st.caption(desc)

        # ── Render active chat response ───────
        active = st.session_state.active_chat
        if active and active in mock_responses:
            st.markdown(
                f'<div class="assistant-msg">{mock_responses[active]}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── New Screening Button ──────────────
        if st.button("🔄 Start New Screening", type="secondary",
                      use_container_width=True, key="new_screening"):
            st.session_state.form_key_counter = st.session_state.get('form_key_counter', 0) + 1
            for k, v in _defaults.items():
                if k != 'form_key_counter':
                    st.session_state[k] = v
            st.session_state.scroll_to_top = True
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown(
        '<p class="rg-hero-title" style="font-size: 2.6rem;">About<br>RenalGuard AI.</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="rg-hero-sub">'
        'An AI-powered clinical decision support system for early detection of '
        'Chronic Kidney Disease — built to save lives through precision screening.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### 🚨 The Problem")

    problem_tiles = [
        ("850M+", "Affected Globally",  "People with kidney disease worldwide"),
        ("90%",   "Undiagnosed",        "Patients unaware until late stages"),
        ("50%",   "Preventable",        "Cases stoppable with early detection"),
        ("$84B+", "Annual Cost",        "Healthcare burden in the USA alone"),
    ]
    cols = st.columns(4)
    for col, (val, lbl, sub) in zip(cols, problem_tiles):
        col.markdown(
            f'<div class="rg-metric">'
            f'<div class="rg-metric-val">{val}</div>'
            f'<div class="rg-metric-lbl">{lbl}</div>'
            f'<div class="rg-metric-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("")
    st.markdown(
        '<div class="glass-card">'
        '<p style="margin: 0; line-height: 1.7; color: #334155;">'
        'Chronic Kidney Disease (CKD) is one of the most underdiagnosed global '
        'health threats. Primary care physicians face challenges interpreting '
        'multiple complex biomarkers, identifying early risk patterns, and '
        'providing clear patient explanations. '
        '<strong>Early detection can prevent progression in up to 50% of cases'
        '</strong>, but requires data-driven screening tools. '
        'RenalGuard AI addresses this gap.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### 💡 Our Solution")

    st.markdown(
        '<div class="glass-card">'
        '<p style="margin: 0; line-height: 1.7; color: #334155;">'
        'RenalGuard AI combines <strong>machine learning</strong>, '
        '<strong>clinical guidelines</strong>, and '
        '<strong>explainable AI</strong> to help healthcare providers detect '
        'CKD risk early, understand why a patient is high risk, generate '
        'professional clinical screening reports, and provide actionable '
        'patient guidance — all by analyzing <strong>24 clinical biomarkers'
        '</strong> for a transparent risk assessment.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### ⚙️ How the System Works")

    steps = [
        ("1️⃣", "Data Input",
         "The clinician enters patient biomarkers — blood pressure, serum "
         "creatinine, glucose levels, urinalysis, and more."),
        ("2️⃣", "AI Risk Detection",
         "A stacked ML ensemble (XGBoost + LightGBM) analyzes the biomarkers "
         "to predict CKD probability with high accuracy."),
        ("3️⃣", "Clinical Staging",
         "The system calculates eGFR using the CKD-EPI 2021 equation and "
         "determines CKD stage per KDIGO guidelines."),
        ("4️⃣", "Explainable AI",
         "SHAP (SHapley Additive Explanations) identifies which biomarkers "
         "most influenced the prediction — in plain English."),
        ("5️⃣", "Clinical Reporting",
         "A professional PDF screening report is generated instantly for "
         "documentation or patient communication."),
    ]

    for icon, title, desc in steps:
        st.markdown(
            f'<div class="glass-card" style="padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;">'
            f'<div style="display: flex; align-items: flex-start; gap: 1rem;">'
            f'<span style="font-size: 1.8rem; line-height: 1;">{icon}</span>'
            f'<div>'
            f'<h4 style="margin: 0 0 0.3rem 0; font-size: 1rem; font-weight: 700;">{title}</h4>'
            f'<p style="margin: 0; font-size: 0.9rem; color: #64748B; line-height: 1.6;">{desc}</p>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### 🚀 Key Features")

    features = [
        ("🤖", "AI Ensemble Model",
         "Combines XGBoost and LightGBM in a stacking ensemble with 5-fold CV for robust CKD risk prediction."),
        ("🔍", "Explainable AI",
         "SHAP-powered explanations translated into plain English — no black boxes, full transparency."),
        ("📊", "24-Biomarker Analysis",
         "Comprehensive evaluation of blood chemistry, urinalysis, vitals, and clinical history."),
        ("📄", "Automated PDF Reports",
         "Professional clinical screening reports with lab values, staging, and KDIGO-aligned recommendations."),
        ("💬", "AI Medical Assistant",
         "Context-aware assistant providing stage-specific dietary guidance and follow-up recommendations."),
        ("🛡️", "Clinical Safety",
         "Input validation, range checking, and clear disclaimers — designed for responsible clinical use."),
    ]

    rows = [features[i:i + 3] for i in range(0, len(features), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, row):
            col.markdown(
                f'<div class="rg-feature" style="min-height: 160px;">'
                f'<h4>{icon} {title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("#### 🧠 Technology Stack")

    tech = [
        ("⚡", "Machine Learning",
         "XGBoost • LightGBM • Scikit-learn • Stacking Ensemble • 5-Fold Cross-Validation"),
        ("🔎", "Explainability",
         "SHAP • TreeExplainer • Feature Attribution • Plain-English Translation"),
        ("🖥️", "Application",
         "Python • Streamlit • Pandas • NumPy • FPDF2 • CKD-EPI 2021"),
    ]

    cols = st.columns(3)
    for col, (icon, title, items) in zip(cols, tech):
        col.markdown(
            f'<div class="rg-metric" style="min-height: 140px; text-align: left; padding: 1.2rem;">'
            f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>'
            f'<div style="font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 0.4rem;">{title}</div>'
            f'<div style="font-size: 0.82rem; color: #64748B; line-height: 1.6;">{items}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### ⚠️ Clinical Disclaimer")

    st.markdown(
        '<div class="glass-card" style="border-left: 4px solid #F59E0B; '
        'background: linear-gradient(135deg, #FFFBEB 0%, rgba(255,255,255,0.9) 100%);">'
        '<p style="margin: 0; line-height: 1.7; color: #92400E; font-size: 0.92rem;">'
        '<strong>⚠️ Important:</strong> RenalGuard AI is designed as a '
        '<strong>clinical decision support tool</strong>. It is '
        '<strong>not</strong> intended to replace professional medical '
        'diagnosis. All predictions are probabilistic and should be '
        'interpreted by qualified healthcare professionals. The AI model '
        'was trained on the UCI Chronic Kidney Disease dataset and uses '
        'the CKD-EPI 2021 equation for eGFR estimation.'
        '</p></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align: center; padding: 2rem 0 1rem 0;">'
        '<p style="font-size: 0.78rem; color: #94A3B8; letter-spacing: 0.04em;">'
        'Built with ❤️ for better kidney health outcomes<br>'
        'RenalGuard AI v2.0 • Clinical Decision Support System'
        '</p></div>',
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    render_nav()
    page = st.session_state.active_page
    if   page == 0: page_home()
    elif page == 1: page_screening()
    elif page == 2: page_about()


if __name__ == '__main__':
    main()