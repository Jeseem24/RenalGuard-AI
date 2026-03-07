"""
RenalGuard AI - Data Preprocessing Module
Handles UCI CKD Dataset loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from typing import Tuple, Dict, Any


class CKDDataPreprocessor:
    """
    Preprocessor for UCI Chronic Kidney Disease Dataset
    Handles:
    - Missing value imputation
    - Categorical encoding
    - Feature engineering (eGFR calculation)
    - CKD Stage label creation
    """
    
    # Feature definitions with normal ranges
    FEATURE_INFO = {
        'age': {'name': 'Age', 'unit': 'years', 'normal_range': None, 'type': 'numerical'},
        'bp': {'name': 'Blood Pressure', 'unit': 'mm/Hg', 'normal_range': (80, 120), 'type': 'numerical'},
        'sg': {'name': 'Specific Gravity', 'unit': '', 'normal_range': (1.005, 1.025), 'type': 'numerical'},
        'al': {'name': 'Albumin', 'unit': '', 'normal_range': (0, 0), 'type': 'ordinal'},
        'su': {'name': 'Sugar', 'unit': '', 'normal_range': (0, 0), 'type': 'ordinal'},
        'rbc': {'name': 'Red Blood Cells', 'unit': '', 'normal_range': 'normal', 'type': 'categorical'},
        'pc': {'name': 'Pus Cell', 'unit': '', 'normal_range': 'normal', 'type': 'categorical'},
        'pcc': {'name': 'Pus Cell Clumps', 'unit': '', 'normal_range': 'notpresent', 'type': 'categorical'},
        'ba': {'name': 'Bacteria', 'unit': '', 'normal_range': 'notpresent', 'type': 'categorical'},
        'bgr': {'name': 'Blood Glucose Random', 'unit': 'mg/dL', 'normal_range': (70, 140), 'type': 'numerical'},
        'bu': {'name': 'Blood Urea', 'unit': 'mg/dL', 'normal_range': (7, 20), 'type': 'numerical'},
        'sc': {'name': 'Serum Creatinine', 'unit': 'mg/dL', 'normal_range': (0.6, 1.2), 'type': 'numerical'},
        'sod': {'name': 'Sodium', 'unit': 'mEq/L', 'normal_range': (136, 145), 'type': 'numerical'},
        'pot': {'name': 'Potassium', 'unit': 'mEq/L', 'normal_range': (3.5, 5.0), 'type': 'numerical'},
        'hemo': {'name': 'Hemoglobin', 'unit': 'gms', 'normal_range': (12, 17), 'type': 'numerical'},
        'pcv': {'name': 'Packed Cell Volume', 'unit': '%', 'normal_range': (36, 50), 'type': 'numerical'},
        'wc': {'name': 'White Blood Cell Count', 'unit': 'cells/cumm', 'normal_range': (4500, 11000), 'type': 'numerical'},
        'rc': {'name': 'Red Blood Cell Count', 'unit': 'millions/cmm', 'normal_range': (4.5, 5.5), 'type': 'numerical'},
        'htn': {'name': 'Hypertension', 'unit': '', 'normal_range': 'no', 'type': 'binary'},
        'dm': {'name': 'Diabetes Mellitus', 'unit': '', 'normal_range': 'no', 'type': 'binary'},
        'cad': {'name': 'Coronary Artery Disease', 'unit': '', 'normal_range': 'no', 'type': 'binary'},
        'appet': {'name': 'Appetite', 'unit': '', 'normal_range': 'good', 'type': 'binary'},
        'pe': {'name': 'Pedal Edema', 'unit': '', 'normal_range': 'no', 'type': 'binary'},
        'ane': {'name': 'Anemia', 'unit': '', 'normal_range': 'no', 'type': 'binary'},
    }
    
    NUMERICAL_FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    CATEGORICAL_FEATURES = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    
    def __init__(self):
        self.label_encoders = {}
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.feature_names = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the UCI CKD dataset"""
        # Try different possible formats
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_csv(filepath, na_values=['?', '\t?', ''])
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Rename target column if needed
        if 'classification' in df.columns:
            df = df.rename(columns={'classification': 'class'})
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset - handle special characters, whitespace, etc."""
        df = df.copy()
        
        # Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        
        # Handle tab characters in target column
        if 'class' in df.columns:
            df['class'] = df['class'].astype(str).str.replace('\t', '').str.strip()
            df['class'] = df['class'].replace({'ckd': 'ckd', 'notckd': 'notckd', 'ckdt': 'ckd'})
        
        # Handle yes/no columns with tab characters
        for col in ['htn', 'dm', 'cad', 'pe', 'ane']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('\t', '').str.strip()
                df[col] = df[col].replace({'yes': 'yes', 'no': 'no', '\tyes': 'yes', '\tno': 'no'})
        
        # Handle appetite column
        if 'appet' in df.columns:
            df['appet'] = df['appet'].astype(str).str.replace('\t', '').str.strip()
        
        return df
    
    def convert_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numerical columns stored as strings to float"""
        df = df.copy()
        
        for col in self.NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def calculate_egfr(self, creatinine: float, age: float) -> float:
        """
        Calculate eGFR using Cockcroft-Gault formula (simplified)
        eGFR = 186 × (Serum Creatinine)^(-1.154) × (Age)^(-0.203)
        """
        if pd.isna(creatinine) or pd.isna(age) or creatinine <= 0:
            return np.nan
        
        egfr = 186 * (creatinine ** -1.154) * (age ** -0.203)
        return round(egfr, 2)
    
    def classify_ckd_stage(self, egfr: float) -> int:
        """
        Classify CKD stage based on eGFR
        Stage 1: >= 90
        Stage 2: 60-89
        Stage 3: 30-59
        Stage 4: 15-29
        Stage 5: < 15
        """
        if pd.isna(egfr):
            return 0  # Unknown
        
        if egfr >= 90:
            return 1
        elif egfr >= 60:
            return 2
        elif egfr >= 30:
            return 3
        elif egfr >= 15:
            return 4
        else:
            return 5
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features including eGFR and CKD stage"""
        df = df.copy()
        
        # Calculate eGFR
        df['egfr'] = df.apply(
            lambda row: self.calculate_egfr(row.get('sc', np.nan), row.get('age', np.nan)),
            axis=1
        )
        
        # Classify CKD stage
        df['ckd_stage'] = df['egfr'].apply(self.classify_ckd_stage)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values using imputation"""
        df = df.copy()
        imp_info = {}
        
        # Numerical imputation
        num_cols = [col for col in self.NUMERICAL_FEATURES if col in df.columns]
        if num_cols:
            if fit:
                self.numerical_imputer = SimpleImputer(strategy='median')
                df[num_cols] = self.numerical_imputer.fit_transform(df[num_cols])
            elif self.numerical_imputer is not None:
                df[num_cols] = self.numerical_imputer.transform(df[num_cols])
            imp_info['numerical'] = {col: 'median' for col in num_cols}
        
        # Categorical imputation
        cat_cols = [col for col in self.CATEGORICAL_FEATURES if col in df.columns]
        if cat_cols:
            if fit:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = self.categorical_imputer.fit_transform(df[cat_cols])
            elif self.categorical_imputer is not None:
                df[cat_cols] = self.categorical_imputer.transform(df[cat_cols])
            imp_info['categorical'] = {col: 'most_frequent' for col in cat_cols}
        
        # Handle engineered features (egfr, ckd_stage) - fill with median/mode
        if 'egfr' in df.columns:
            egfr_median = df['egfr'].median()
            df['egfr'] = df['egfr'].fillna(egfr_median if not pd.isna(egfr_median) else 90)
        
        if 'ckd_stage' in df.columns:
            stage_mode = df['ckd_stage'].mode()
            df['ckd_stage'] = df['ckd_stage'].fillna(stage_mode.iloc[0] if len(stage_mode) > 0 else 2)
        
        # Final check - fill any remaining NaN with 0
        df = df.fillna(0)
        
        return df, imp_info
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        # Handle unseen labels
                        df[col] = df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        return df
    
    def encode_target(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Encode target variable"""
        df = df.copy()
        encoding_map = {}
        
        if 'class' in df.columns:
            if fit:
                le = LabelEncoder()
                df['class'] = le.fit_transform(df['class'].astype(str))
                self.label_encoders['class'] = le
                encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
            else:
                le = self.label_encoders.get('class')
                if le:
                    df['class'] = le.transform(df['class'].astype(str))
        
        return df, encoding_map
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Full preprocessing pipeline - fit and transform"""
        info = {}
        
        # Clean data
        df = self.clean_data(df)
        info['cleaning'] = 'Removed whitespace, tabs, special characters'
        
        # Convert numerical columns
        df = self.convert_numerical_columns(df)
        info['conversion'] = 'Converted numerical columns from string'
        
        # Engineer features
        df = self.engineer_features(df)
        info['feature_engineering'] = 'Added eGFR and ckd_stage features'
        
        # Handle missing values
        df, imp_info = self.handle_missing_values(df, fit=True)
        info['imputation'] = imp_info
        
        # Encode categorical features
        df = self.encode_categorical(df, fit=True)
        info['encoding'] = 'Label encoded categorical features'
        
        # Encode target
        df, target_map = self.encode_target(df, fit=True)
        info['target_encoding'] = target_map
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['class', 'ckd_stage']]
        
        return df, info
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        df = self.clean_data(df)
        df = self.convert_numerical_columns(df)
        df = self.engineer_features(df)
        df, _ = self.handle_missing_values(df, fit=False)
        df = self.encode_categorical(df, fit=False)
        df, _ = self.encode_target(df, fit=False)
        return df
    
    def save(self, filepath: str):
        """Save preprocessor to file"""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'CKDDataPreprocessor':
        """Load preprocessor from file"""
        return joblib.load(filepath)
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get information about a specific feature"""
        return self.FEATURE_INFO.get(feature_name, {})
    
    def get_status(self, feature_name: str, value: float) -> str:
        """Get the status (Normal/High/Low) for a feature value"""
        info = self.FEATURE_INFO.get(feature_name)
        if not info or not info.get('normal_range'):
            return 'N/A'
        
        normal_range = info['normal_range']
        
        if isinstance(normal_range, tuple):
            low, high = normal_range
            if value < low:
                return 'LOW'
            elif value > high:
                return 'HIGH'
            else:
                return 'NORMAL'
        
        return 'N/A'


def create_sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for testing when real data is not available"""
    np.random.seed(42)
    n_samples = 400
    
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'bp': np.random.randint(60, 180, n_samples),
        'sg': np.round(np.random.uniform(1.005, 1.025, n_samples), 3),
        'al': np.random.randint(0, 5, n_samples),
        'su': np.random.randint(0, 5, n_samples),
        'rbc': np.random.choice(['normal', 'abnormal'], n_samples),
        'pc': np.random.choice(['normal', 'abnormal'], n_samples),
        'pcc': np.random.choice(['present', 'notpresent'], n_samples),
        'ba': np.random.choice(['present', 'notpresent'], n_samples),
        'bgr': np.random.randint(70, 200, n_samples),
        'bu': np.random.randint(10, 100, n_samples),
        'sc': np.round(np.random.uniform(0.5, 5.0, n_samples), 2),
        'sod': np.random.randint(130, 150, n_samples),
        'pot': np.round(np.random.uniform(3.0, 6.0, n_samples), 1),
        'hemo': np.round(np.random.uniform(8, 17, n_samples), 1),
        'pcv': np.random.randint(25, 55, n_samples),
        'wc': np.random.randint(3000, 15000, n_samples),
        'rc': np.round(np.random.uniform(3.0, 7.0, n_samples), 1),
        'htn': np.random.choice(['yes', 'no'], n_samples),
        'dm': np.random.choice(['yes', 'no'], n_samples),
        'cad': np.random.choice(['yes', 'no'], n_samples),
        'appet': np.random.choice(['good', 'poor'], n_samples),
        'pe': np.random.choice(['yes', 'no'], n_samples),
        'ane': np.random.choice(['yes', 'no'], n_samples),
        'class': np.random.choice(['ckd', 'notckd'], n_samples, p=[0.625, 0.375])
    }
    
    # Add some correlation between features and target
    df = pd.DataFrame(data)
    
    # Make CKD patients have higher creatinine, urea, lower hemoglobin
    ckd_mask = df['class'] == 'ckd'
    df.loc[ckd_mask, 'sc'] = np.clip(df.loc[ckd_mask, 'sc'] * 1.5, 0.5, 10)
    df.loc[ckd_mask, 'bu'] = np.clip(df.loc[ckd_mask, 'bu'] * 1.3, 10, 150)
    df.loc[ckd_mask, 'hemo'] = np.clip(df.loc[ckd_mask, 'hemo'] * 0.8, 6, 14)
    
    # Add missing values randomly (about 10%)
    for col in df.columns:
        if col != 'class':
            mask = np.random.random(n_samples) < 0.1
            df.loc[mask, col] = np.nan
    
    return df


if __name__ == '__main__':
    # Test the preprocessor
    preprocessor = CKDDataPreprocessor()
    
    # Create sample data for testing
    df = create_sample_dataset()
    print("Sample dataset created with shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Preprocess
    df_processed, info = preprocessor.fit_transform(df)
    print("\nProcessed dataset shape:", df_processed.shape)
    print("\nPreprocessing info:", info)
    
    print("\nPreprocessor test completed successfully!")
