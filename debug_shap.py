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

print("Loading models...")
det = joblib.load('reports/models/ckd_detector.joblib')
model = det['model']
features = det['feature_names']

pre = CKDDataPreprocessor.load('reports/models/preprocessor.joblib')

pt = {
    'age': 48, 'bp': 80, 'bgr': 117, 'sc': 3.0, 'pot': 4.0, 'pcv': 43, 'rc': 5.0,
    'bu': 31.0, 'sod': 137, 'hemo': 14.1, 'wc': 8000, 'sg': 1.02, 'su': 0, 'al': 5,
    'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'present',
    'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
}

print("Transforming data...")
df_pt = pd.DataFrame([pt])
df_tx = pre.transform(df_pt)
for col in features:
    if col not in df_tx.columns:
        df_tx[col] = 0
X_pt = df_tx[features]

print("Initializing explainer...")
exp = SHAPExplainer(model, features)
bg_data = pre.transform(create_sample_dataset())
for c in features:
    if c not in bg_data.columns:
        bg_data[c] = 0
bg_data = bg_data[features].head(80)

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
