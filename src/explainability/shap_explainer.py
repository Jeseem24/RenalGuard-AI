"""
RenalGuard AI - SHAP Explainability Module
Provides interpretable AI explanations for predictions
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, List, Tuple, Optional, Any
import io
import base64


class SHAPExplainer:
    """
    SHAP-based explainability for CKD predictions
    Generates:
    - Waterfall plots for individual predictions
    - Summary plots for global feature importance
    - Human-readable explanations
    """
    
    # Feature descriptions for human-readable explanations
    FEATURE_DESCRIPTIONS = {
        'age': {
            'name': 'Age',
            'description': 'Patient age in years',
            'normal_range': 'N/A',
            'high_implication': 'Older age increases CKD risk',
            'low_implication': 'Younger age associated with lower CKD risk'
        },
        'bp': {
            'name': 'Blood Pressure',
            'description': 'Blood pressure in mm/Hg',
            'normal_range': '80-120 mm/Hg',
            'high_implication': 'High blood pressure damages kidney blood vessels',
            'low_implication': 'Normal blood pressure is protective'
        },
        'sg': {
            'name': 'Specific Gravity',
            'description': 'Urine concentration indicator',
            'normal_range': '1.005-1.025',
            'high_implication': 'Abnormal concentration may indicate kidney issues',
            'low_implication': 'Normal specific gravity'
        },
        'al': {
            'name': 'Albumin',
            'description': 'Albumin level in urine',
            'normal_range': '0 (none)',
            'high_implication': 'Presence of albumin indicates kidney damage',
            'low_implication': 'No albumin in urine is normal'
        },
        'su': {
            'name': 'Sugar',
            'description': 'Sugar level in urine',
            'normal_range': '0 (none)',
            'high_implication': 'Sugar in urine may indicate diabetes',
            'low_implication': 'No sugar in urine is normal'
        },
        'rbc': {
            'name': 'Red Blood Cells',
            'description': 'RBC presence in urine',
            'normal_range': 'Normal',
            'high_implication': 'Abnormal RBC may indicate kidney damage',
            'low_implication': 'Normal RBC count'
        },
        'pc': {
            'name': 'Pus Cells',
            'description': 'Pus cell presence in urine',
            'normal_range': 'Normal',
            'high_implication': 'Abnormal pus cells may indicate infection',
            'low_implication': 'Normal pus cell count'
        },
        'bgr': {
            'name': 'Blood Glucose Random',
            'description': 'Random blood glucose level',
            'normal_range': '70-140 mg/dL',
            'high_implication': 'High glucose may indicate diabetes, a CKD risk factor',
            'low_implication': 'Normal blood glucose'
        },
        'bu': {
            'name': 'Blood Urea',
            'description': 'Blood urea nitrogen level',
            'normal_range': '7-20 mg/dL',
            'high_implication': 'Elevated urea indicates reduced kidney filtration',
            'low_implication': 'Normal blood urea level'
        },
        'sc': {
            'name': 'Serum Creatinine',
            'description': 'Creatinine level in blood',
            'normal_range': '0.6-1.2 mg/dL',
            'high_implication': 'Elevated creatinine indicates reduced kidney function',
            'low_implication': 'Normal creatinine level'
        },
        'sod': {
            'name': 'Sodium',
            'description': 'Sodium level in blood',
            'normal_range': '136-145 mEq/L',
            'high_implication': 'Abnormal sodium may indicate electrolyte imbalance',
            'low_implication': 'Normal sodium level'
        },
        'pot': {
            'name': 'Potassium',
            'description': 'Potassium level in blood',
            'normal_range': '3.5-5.0 mEq/L',
            'high_implication': 'Abnormal potassium can affect heart and kidney function',
            'low_implication': 'Normal potassium level'
        },
        'hemo': {
            'name': 'Hemoglobin',
            'description': 'Hemoglobin level in blood',
            'normal_range': '12-17 gms',
            'high_implication': 'High hemoglobin (rare in CKD)',
            'low_implication': 'Low hemoglobin (anemia) is common in CKD'
        },
        'pcv': {
            'name': 'Packed Cell Volume',
            'description': 'Percentage of red blood cells in blood',
            'normal_range': '36-50%',
            'high_implication': 'Abnormal PCV may indicate dehydration or other issues',
            'low_implication': 'Low PCV indicates anemia, common in CKD'
        },
        'wc': {
            'name': 'White Blood Cell Count',
            'description': 'WBC count in blood',
            'normal_range': '4500-11000 cells/cumm',
            'high_implication': 'High WBC may indicate infection or inflammation',
            'low_implication': 'Normal WBC count'
        },
        'rc': {
            'name': 'Red Blood Cell Count',
            'description': 'RBC count in blood',
            'normal_range': '4.5-5.5 millions/cmm',
            'high_implication': 'Abnormal RBC count',
            'low_implication': 'Normal RBC count'
        },
        'htn': {
            'name': 'Hypertension',
            'description': 'Whether patient has hypertension',
            'normal_range': 'No',
            'high_implication': 'Hypertension is a major CKD risk factor',
            'low_implication': 'No hypertension is protective'
        },
        'dm': {
            'name': 'Diabetes Mellitus',
            'description': 'Whether patient has diabetes',
            'normal_range': 'No',
            'high_implication': 'Diabetes is the #1 cause of CKD',
            'low_implication': 'No diabetes reduces CKD risk'
        },
        'cad': {
            'name': 'Coronary Artery Disease',
            'description': 'Whether patient has CAD',
            'normal_range': 'No',
            'high_implication': 'Heart disease increases CKD risk',
            'low_implication': 'No CAD is protective'
        },
        'appet': {
            'name': 'Appetite',
            'description': 'Patient appetite status',
            'normal_range': 'Good',
            'high_implication': 'Poor appetite may indicate advanced CKD',
            'low_implication': 'Good appetite is normal'
        },
        'pe': {
            'name': 'Pedal Edema',
            'description': 'Swelling in legs/feet',
            'normal_range': 'No',
            'high_implication': 'Edema indicates fluid retention from kidney dysfunction',
            'low_implication': 'No edema is normal'
        },
        'ane': {
            'name': 'Anemia',
            'description': 'Whether patient has anemia',
            'normal_range': 'No',
            'high_implication': 'Anemia is common in CKD due to reduced EPO production',
            'low_implication': 'No anemia is normal'
        },
        'egfr': {
            'name': 'eGFR',
            'description': 'Estimated Glomerular Filtration Rate',
            'normal_range': '>=90 mL/min/1.73m²',
            'high_implication': 'Low eGFR directly indicates kidney dysfunction',
            'low_implication': 'Normal eGFR indicates good kidney function'
        }
    }
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model (XGBoost, LightGBM, or sklearn)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def fit(self, X: pd.DataFrame, sample_size: int = 100):
        """
        Fit the SHAP explainer
        
        Args:
            X: Training data for background distribution
            sample_size: Number of samples to use for background
        """
        # Sample background data
        if len(X) > sample_size:
            background = shap.sample(X, sample_size)
        else:
            background = X
        
        # Create explainer based on model type
        try:
            # Try TreeExplainer first (fastest for tree-based models)
            self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fall back to KernelExplainer (slower but works for any model)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        
        print("SHAP explainer initialized successfully")
    
    def explain_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction
        
        Args:
            X: Single sample dataframe
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification, take the positive class values
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create explanation dictionary
        feature_contributions = []
        for i, feature in enumerate(self.feature_names):
            contribution = {
                'feature': feature,
                'shap_value': float(shap_values[i]),
                'feature_value': float(X.iloc[0, i]),
                'contribution_pct': 0  # Will be calculated below
            }
            feature_contributions.append(contribution)
        
        # Calculate percentage contributions (absolute values)
        total_abs_shap = sum(abs(c['shap_value']) for c in feature_contributions)
        if total_abs_shap > 0:
            for c in feature_contributions:
                c['contribution_pct'] = abs(c['shap_value']) / total_abs_shap * 100
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Get top risk factors
        top_risk_factors = [c for c in feature_contributions if c['shap_value'] > 0][:5]
        top_protective_factors = [c for c in feature_contributions if c['shap_value'] < 0][:5]
        
        return {
            'feature_contributions': feature_contributions,
            'top_risk_factors': top_risk_factors,
            'top_protective_factors': top_protective_factors,
            'shap_values': shap_values.tolist()
        }
    
    def generate_waterfall_plot(self, X: pd.DataFrame, output_path: str = None) -> Optional[str]:
        """
        Generate SHAP waterfall plot for a prediction
        
        Args:
            X: Single sample dataframe
            output_path: Path to save the plot (optional)
            
        Returns:
            Base64 encoded image if no output_path, else None
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        
        # Map feature names to display names using FEATURE_DESCRIPTIONS
        display_names = [self.FEATURE_DESCRIPTIONS.get(f, {}).get('name', f.upper()) for f in self.feature_names]
        
        # Create explanation object for waterfall plot
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=X.iloc[0].values,
            feature_names=display_names
        )
        
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title('Patient-Specific Risk Drivers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return None
        else:
            # Return base64 encoded image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
    
    def generate_summary_plot(self, X: pd.DataFrame, output_path: str = None) -> Optional[str]:
        """
        Generate SHAP summary plot showing global feature importance
        
        Args:
            X: Dataset to explain
            output_path: Path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        plt.figure(figsize=(12, 10))
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
    
    def generate_human_readable_explanation(self, explanation: Dict, patient_values: Dict) -> str:
        """
        Generate a human-readable explanation text
        
        Args:
            explanation: Output from explain_prediction()
            patient_values: Dictionary of patient test values
            
        Returns:
            Human-readable explanation string
        """
        top_factors = explanation['top_risk_factors'][:5]
        
        if not top_factors:
            return "No significant risk factors identified. The patient's kidney health appears normal."
        
        explanation_text = "CKD Risk Assessment Explanation:\n\n"
        explanation_text += "The AI identified the following factors contributing to the CKD risk:\n\n"
        
        for i, factor in enumerate(top_factors, 1):
            feature = factor['feature']
            value = factor['feature_value']
            contribution = factor['contribution_pct']
            
            # Get feature info
            info = self.FEATURE_DESCRIPTIONS.get(feature, {})
            feature_name = info.get('name', feature)
            normal_range = info.get('normal_range', 'N/A')
            implication = info.get('high_implication', 'Contributes to risk')
            
            explanation_text += f"{i}. {feature_name} ({value})\n"
            explanation_text += f"   • Contribution to risk: {contribution:.1f}%\n"
            explanation_text += f"   • Normal range: {normal_range}\n"
            explanation_text += f"   • {implication}\n\n"
        
        return explanation_text
    
    def get_clinical_interpretation(self, explanation: Dict) -> List[str]:
        """
        Get clinical interpretation bullet points
        
        Args:
            explanation: Output from explain_prediction()
            
        Returns:
            List of clinical interpretation strings
        """
        interpretations = []
        top_factors = explanation['top_risk_factors'][:5]
        
        for factor in top_factors:
            feature = factor['feature']
            value = factor['feature_value']
            contribution = factor['contribution_pct']
            
            info = self.FEATURE_DESCRIPTIONS.get(feature, {})
            feature_name = info.get('name', feature)
            normal_range = info.get('normal_range', 'N/A')
            
            # Determine status
            status = "⚠️ ELEVATED" if factor['shap_value'] > 0 else "✅ NORMAL"
            
            interpretations.append(
                f"{feature_name}: {value} (Normal: {normal_range}) - {status} [{contribution:.1f}% contribution]"
            )
        
        return interpretations


if __name__ == '__main__':
    print("SHAP Explainer module loaded successfully")
    print("This module provides explainability for CKD predictions")
