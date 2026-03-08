import sys
import os
import pandas as pd
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ckd_detector import CKDDetector
from src.preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
from src.explainability.shap_explainer import SHAPExplainer
import joblib

def load_models():
    mp = 'reports/models'
    dp, sp, pp = f"{mp}/ckd_detector.joblib", f"{mp}/stage_classifier.joblib", f"{mp}/preprocessor.joblib"
    
    md = joblib.load(dp)
    det = CKDDetector()
    det.best_model, det.best_model_name = md['model'], md['model_name']
    det.feature_names, det.metrics = md['feature_names'], md['metrics']

    pre = CKDDataPreprocessor.load(pp)
    return det, pre, det.feature_names

def get_shap_background_data():
    ds = create_sample_dataset()
    bg_pre = CKDDataPreprocessor(use_knn_imputer=True)
    dsp, _ = bg_pre.fit_transform(ds)
    fc = [c for c in dsp.columns if c not in ['class', 'ckd_stage']]
    return dsp[fc]

print("Loading models...")
detector, preprocessor, feature_cols = load_models()

pt = {
    'age': 48, 'bp': 80, 'bgr': 117, 'sc': 3.0, 'pot': 4.0, 'pcv': 43, 'rc': 5.0,
    'bu': 31.0, 'sod': 137, 'hemo': 14.1, 'wc': 8000, 'sg': 1.02, 'su': 0, 'al': 5,
    'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'present',
    'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
}

print("Transforming data...")
df_pt = pd.DataFrame([pt])
df_tx = preprocessor.transform(df_pt)
for col in feature_cols:
    if col not in df_tx.columns:
        df_tx[col] = 0
X_pt = df_tx[feature_cols]

print("Initializing explainer...")
exp = SHAPExplainer(detector.best_model, feature_cols)
bg_data = get_shap_background_data()
for c in feature_cols:
    if c not in bg_data.columns:
        bg_data[c] = 0
bg_data = bg_data[feature_cols]

# Just like in main.py
exp.fit(bg_data, sample_size=80)

print("Explaining prediction...")
try:
    exp_data = exp.explain_prediction(X_pt)
    print("explain_prediction succeeded!")
except Exception as e:
    print(f"explain_prediction FAILED:")
    traceback.print_exc()

print("Generating waterfall plot...")
try:
    b64 = exp.generate_waterfall_plot(X_pt)
    print("generate_waterfall_plot succeeded!")
except Exception as e:
    print(f"generate_waterfall_plot FAILED:")
    traceback.print_exc()

import shap
print("SHAP version:", shap.__version__)
