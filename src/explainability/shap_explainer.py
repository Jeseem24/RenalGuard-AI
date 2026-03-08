"""
RenalGuard AI - SHAP Explainability Module
Generates interpretable, plain-English explanations for CKD predictions
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import io
import base64
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP-based explainability for CKD predictions.
    Generates waterfall plots, summary plots, and human-readable clinical explanations.
    """

    FEATURE_CLINICAL = {
        'age':  {'name': 'Age',                    'normal': None,         'risk_msg': 'Advanced age increases CKD risk due to natural kidney decline.',                          'protect_msg': 'Younger age is protective for kidney function.'},
        'bp':   {'name': 'Blood Pressure',          'normal': '80–120 mmHg','risk_msg': 'Elevated blood pressure damages glomerular capillaries over time.',                      'protect_msg': 'Normal blood pressure preserves kidney microvasculature.'},
        'sg':   {'name': 'Specific Gravity',        'normal': '1.005–1.025','risk_msg': 'Abnormal urine concentration suggests impaired renal tubular function.',                  'protect_msg': 'Normal specific gravity indicates healthy tubular function.'},
        'al':   {'name': 'Albumin (Urine)',         'normal': '0 (absent)', 'risk_msg': 'Albuminuria is a hallmark of glomerular damage and early CKD.',                           'protect_msg': 'Absence of albumin in urine is a key healthy indicator.'},
        'su':   {'name': 'Sugar (Urine)',           'normal': '0 (absent)', 'risk_msg': 'Glycosuria may indicate uncontrolled diabetes — the #1 cause of CKD.',                    'protect_msg': 'No sugar in urine suggests adequate glycemic control.'},
        'rbc':  {'name': 'Red Blood Cells (Urine)', 'normal': 'Normal',    'risk_msg': 'Hematuria can indicate glomerulonephritis or kidney stones.',                              'protect_msg': 'Normal urinary RBC rules out active glomerular bleeding.'},
        'pc':   {'name': 'Pus Cells',               'normal': 'Normal',    'risk_msg': 'Elevated pus cells suggest urinary tract infection or interstitial nephritis.',             'protect_msg': 'Normal pus cell count rules out active urinary infection.'},
        'pcc':  {'name': 'Pus Cell Clumps',          'normal': 'Absent',    'risk_msg': 'Pus cell clumps indicate significant infection or inflammation.',                          'protect_msg': 'Absence of clumps is reassuring.'},
        'ba':   {'name': 'Bacteria',                 'normal': 'Absent',    'risk_msg': 'Bacteriuria indicates active urinary tract infection requiring treatment.',                'protect_msg': 'No bacteria detected is normal.'},
        'bgr':  {'name': 'Blood Glucose',            'normal': '70–140 mg/dL','risk_msg': 'Hyperglycemia causes hyperfiltration and progressive glomerular sclerosis.',             'protect_msg': 'Normal blood glucose reduces diabetic nephropathy risk.'},
        'bu':   {'name': 'Blood Urea',               'normal': '7–20 mg/dL','risk_msg': 'Elevated BUN indicates reduced renal clearance of nitrogenous waste.',                    'protect_msg': 'Normal urea reflects adequate kidney filtration capacity.'},
        'sc':   {'name': 'Serum Creatinine',          'normal': '0.6–1.2 mg/dL','risk_msg': 'Elevated creatinine directly indicates reduced glomerular filtration rate.',           'protect_msg': 'Normal creatinine suggests preserved kidney function.'},
        'sod':  {'name': 'Sodium',                    'normal': '136–145 mEq/L','risk_msg': 'Sodium imbalance may indicate impaired renal tubular regulation.',                    'protect_msg': 'Normal sodium reflects intact electrolyte homeostasis.'},
        'pot':  {'name': 'Potassium',                 'normal': '3.5–5.0 mEq/L','risk_msg': 'Hyperkalemia is dangerous in CKD and indicates impaired renal excretion.',            'protect_msg': 'Normal potassium indicates healthy renal potassium handling.'},
        'hemo': {'name': 'Hemoglobin',                'normal': '12–17 g/dL','risk_msg': 'Low hemoglobin (anemia) is common in CKD due to reduced erythropoietin production.',     'protect_msg': 'Healthy hemoglobin level indicates adequate renal EPO production.'},
        'pcv':  {'name': 'Packed Cell Volume',        'normal': '36–50%',   'risk_msg': 'Low PCV (anemia) correlates with CKD progression.',                                       'protect_msg': 'Normal PCV is a healthy sign.'},
        'wc':   {'name': 'WBC Count',                 'normal': '4500–11000','risk_msg': 'Elevated WBC may indicate chronic inflammation or infection accelerating CKD.',           'protect_msg': 'Normal WBC count rules out active infection.'},
        'rc':   {'name': 'RBC Count',                 'normal': '4.5–5.5 M/cmm','risk_msg': 'Low RBC count reflects renal anemia common in CKD.',                                 'protect_msg': 'Normal RBC count is reassuring.'},
        'htn':  {'name': 'Hypertension',              'normal': 'No',       'risk_msg': 'Hypertension is both a cause and consequence of CKD — tight control is essential.',        'protect_msg': 'Absence of hypertension is a major protective factor.'},
        'dm':   {'name': 'Diabetes Mellitus',         'normal': 'No',       'risk_msg': 'Diabetes is the leading cause of CKD worldwide.',                                         'protect_msg': 'No diabetes significantly reduces CKD risk.'},
        'cad':  {'name': 'Coronary Artery Disease',   'normal': 'No',       'risk_msg': 'Cardiovascular disease shares risk pathways with CKD.',                                   'protect_msg': 'No CAD is protective.'},
        'appet':{'name': 'Appetite',                  'normal': 'Good',     'risk_msg': 'Poor appetite may indicate uremic toxin accumulation in advanced CKD.',                    'protect_msg': 'Good appetite suggests no uremic symptoms.'},
        'pe':   {'name': 'Pedal Edema',               'normal': 'No',       'risk_msg': 'Edema indicates fluid retention from impaired renal excretion.',                           'protect_msg': 'No edema means kidneys are handling fluid balance well.'},
        'ane':  {'name': 'Anemia',                    'normal': 'No',       'risk_msg': 'Anemia in CKD results from decreased erythropoietin and iron deficiency.',                 'protect_msg': 'No anemia is a positive sign for kidney health.'},
        'egfr': {'name': 'eGFR',                      'normal': '≥90 mL/min','risk_msg': 'Low eGFR directly quantifies reduced kidney filtration capacity.',                       'protect_msg': 'Normal eGFR indicates healthy kidney filtration.'},
    }

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._background_set = False

    def fit(self, X: pd.DataFrame, sample_size: int = 100):
        """Fit SHAP explainer with background distribution."""
        bg = shap.sample(X, min(sample_size, len(X))) if len(X) > sample_size else X
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.KernelExplainer(self.model.predict_proba, bg)
        self._background_set = True
    
    def _get_reduced_sv(self, X: pd.DataFrame) -> tuple:
        """Helper to get 1D SHAP values and scalar base value reliably."""
        sv_raw = self.explainer.shap_values(X)
        n_features = len(self.feature_names)
        
        # 1. Force SV into a standard numpy array, flattening lists if necessary
        if isinstance(sv_raw, list):
            sv_raw = np.array(sv_raw)
            if sv_raw.ndim >= 3:
                 # Usually (classes, samples, features) e.g., (2, 1, 24)
                 sv = sv_raw[1] if sv_raw.shape[0] > 1 else sv_raw[0]
            else:
                 sv = sv_raw
        else:
            sv = np.array(sv_raw)
            
        # Strip single-dimensional entries (e.g. 1 sample)
        sv = np.squeeze(sv)
        
        # 2. Robust Feature Extraction
        # We MUST return a 1D array of shape (n_features,)
        if sv.ndim == 0:
            sv = np.zeros(n_features)
        elif sv.ndim == 1:
            if len(sv) == n_features:
                pass # Perfect
            elif len(sv) > n_features:
                sv = sv[:n_features]
            else:
                # Too small, pad with zeros to prevent IndexError
                sv = np.pad(sv, (0, n_features - len(sv)), 'constant')
        elif sv.ndim == 2:
            # Could be (features, classes) or (classes, features)
            if sv.shape[0] == n_features:
                sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]
            elif sv.shape[1] == n_features:
                sv = sv[1, :] if sv.shape[0] > 1 else sv[0, :]
            else:
                # Force resize if shapes are completely unexpected
                sv = sv.flatten()
                if len(sv) >= n_features:
                    sv = sv[:n_features]
                else:
                    sv = np.pad(sv, (0, n_features - len(sv)), 'constant')
        else:
            # 3D+ fallback
            sv = sv.flatten()[:n_features]
            if len(sv) < n_features:
                sv = np.pad(sv, (0, n_features - len(sv)), 'constant')
            
        # 2. Handle base value reduction
        bv = self.explainer.expected_value
        if hasattr(bv, "__len__"):
            bv_arr = np.array(bv).flatten()
            if bv_arr.size > 1:
                bv = bv_arr[1] # Take positive class
            else:
                bv = bv_arr[0]
        elif isinstance(bv, np.ndarray) and bv.ndim == 0:
            bv = float(bv)
            
        return sv, float(bv)

    def explain_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP explanation for a single patient."""
        if self.explainer is None:
            raise RuntimeError("Call fit() first.")

        # Get robustly reduced values
        sv, base_value = self._get_reduced_sv(X)

        total_abs = np.sum(np.abs(sv))
        contributions = []
        for i, feat in enumerate(self.feature_names):
            val = float(sv[i])
            contributions.append({
                'feature': feat,
                'shap_value': val,
                'feature_value': float(X.iloc[0, i]),
                'contribution_pct': (abs(val) / total_abs * 100) if total_abs > 0 else 0,
                'direction': 'risk' if val > 0 else 'protective',
            })

        contributions.sort(key=lambda c: abs(c['shap_value']), reverse=True)

        return {
            'feature_contributions': contributions,
            'top_risk_factors':       [c for c in contributions if c['shap_value'] > 0][:5],
            'top_protective_factors': [c for c in contributions if c['shap_value'] < 0][:5],
            'shap_values': sv.tolist(),
        }

    def generate_plain_english(self, explanation: Dict, risk_level: str) -> str:
        """
        Convert SHAP output into a paragraph a patient or GP can understand.
        This is the "plain English" promise from the README.
        """
        risk_factors = explanation.get('top_risk_factors', [])[:3]
        protect_factors = explanation.get('top_protective_factors', [])[:2]

        if not risk_factors:
            return ("Your biomarker profile does not show significant kidney disease risk factors. "
                    "Continue regular check-ups and maintain a healthy lifestyle.")

        lines = [f"Your overall CKD risk is assessed as **{risk_level}**. Here's why:\n"]

        lines.append("**Key concerns:**")
        for i, f in enumerate(risk_factors, 1):
            info = self.FEATURE_CLINICAL.get(f['feature'], {})
            name = info.get('name', f['feature'].upper())
            msg = info.get('risk_msg', 'This value contributed to your risk.')
            normal = info.get('normal', 'N/A')
            lines.append(
                f"{i}. **{name}** (your value: {f['feature_value']}, normal: {normal}) — "
                f"{msg} This factor contributed {f['contribution_pct']:.0f}% to the risk assessment."
            )

        if protect_factors:
            lines.append("\n**Positive factors working in your favour:**")
            for f in protect_factors:
                info = self.FEATURE_CLINICAL.get(f['feature'], {})
                name = info.get('name', f['feature'].upper())
                msg = info.get('protect_msg', 'This value is protective.')
                lines.append(f"• **{name}** — {msg}")

        lines.append(
            "\n⚠️ *This is an AI-generated explanation for educational purposes. "
            "Please discuss these results with your doctor.*"
        )
        return "\n".join(lines)

    def generate_waterfall_plot(self, X: pd.DataFrame) -> Optional[str]:
        """Generate SHAP waterfall plot, return base64 PNG string."""
        if self.explainer is None:
            raise RuntimeError("Call fit() first.")

        # Get robustly reduced values
        sv, base_value = self._get_reduced_sv(X)

        display_names = [self.FEATURE_CLINICAL.get(f, {}).get('name', f.upper()) for f in self.feature_names]

        explanation_obj = shap.Explanation(
            values=sv, base_values=base_value,
            data=X.iloc[0].values, feature_names=display_names,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(explanation_obj, max_display=12, show=False)
        plt.title('Biomarker Contribution to CKD Risk', fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def get_clinical_bullets(self, explanation: Dict) -> List[str]:
        """Return clinical bullet points for PDF reports."""
        bullets = []
        for f in explanation.get('top_risk_factors', [])[:5]:
            info = self.FEATURE_CLINICAL.get(f['feature'], {})
            name = info.get('name', f['feature'].upper())
            normal = info.get('normal', 'N/A')
            status = "⚠️ CONCERN" if f['shap_value'] > 0 else "✅ OK"
            bullets.append(
                f"{name}: {f['feature_value']} (Normal: {normal}) — "
                f"{status} [{f['contribution_pct']:.0f}% contribution]"
            )
        return bullets