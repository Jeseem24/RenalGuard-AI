"""
RenalGuard AI - Data Preprocessing Module
Handles UCI CKD Dataset loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class CKDDataPreprocessor:
    """
    Preprocessor for UCI Chronic Kidney Disease Dataset
    Handles:
    - Missing value imputation (KNN for numerical, mode for categorical)
    - Categorical encoding
    - Feature engineering (eGFR calculation via CKD-EPI)
    - CKD Stage derivation from eGFR
    - Input validation with medical range checking
    """

    FEATURE_INFO = {
        'age':  {'name': 'Age',                    'unit': 'years',        'range': (2, 120),      'normal': None,         'type': 'numerical'},
        'bp':   {'name': 'Blood Pressure',          'unit': 'mmHg',         'range': (40, 250),     'normal': (80, 120),    'type': 'numerical'},
        'sg':   {'name': 'Specific Gravity',        'unit': '',             'range': (1.000, 1.040),'normal': (1.005, 1.025),'type': 'numerical'},
        'al':   {'name': 'Albumin',                 'unit': '',             'range': (0, 5),        'normal': (0, 0),       'type': 'ordinal'},
        'su':   {'name': 'Sugar',                   'unit': '',             'range': (0, 5),        'normal': (0, 0),       'type': 'ordinal'},
        'rbc':  {'name': 'Red Blood Cells',         'unit': '',             'range': None,          'normal': 'normal',     'type': 'categorical'},
        'pc':   {'name': 'Pus Cell',                'unit': '',             'range': None,          'normal': 'normal',     'type': 'categorical'},
        'pcc':  {'name': 'Pus Cell Clumps',         'unit': '',             'range': None,          'normal': 'notpresent', 'type': 'categorical'},
        'ba':   {'name': 'Bacteria',                'unit': '',             'range': None,          'normal': 'notpresent', 'type': 'categorical'},
        'bgr':  {'name': 'Blood Glucose Random',    'unit': 'mg/dL',        'range': (20, 500),     'normal': (70, 140),    'type': 'numerical'},
        'bu':   {'name': 'Blood Urea',              'unit': 'mg/dL',        'range': (1, 400),      'normal': (7, 20),      'type': 'numerical'},
        'sc':   {'name': 'Serum Creatinine',        'unit': 'mg/dL',        'range': (0.1, 50),     'normal': (0.6, 1.2),   'type': 'numerical'},
        'sod':  {'name': 'Sodium',                  'unit': 'mEq/L',        'range': (100, 180),    'normal': (136, 145),   'type': 'numerical'},
        'pot':  {'name': 'Potassium',               'unit': 'mEq/L',        'range': (2.0, 10.0),   'normal': (3.5, 5.0),   'type': 'numerical'},
        'hemo': {'name': 'Hemoglobin',              'unit': 'g/dL',         'range': (3.0, 20.0),   'normal': (12.0, 17.0), 'type': 'numerical'},
        'pcv':  {'name': 'Packed Cell Volume',       'unit': '%',            'range': (10, 65),      'normal': (36, 50),     'type': 'numerical'},
        'wc':   {'name': 'WBC Count',               'unit': 'cells/cumm',   'range': (2000, 30000), 'normal': (4500, 11000),'type': 'numerical'},
        'rc':   {'name': 'RBC Count',               'unit': 'millions/cmm', 'range': (2.0, 8.0),    'normal': (4.5, 5.5),   'type': 'numerical'},
        'htn':  {'name': 'Hypertension',            'unit': '',             'range': None,          'normal': 'no',         'type': 'binary'},
        'dm':   {'name': 'Diabetes Mellitus',       'unit': '',             'range': None,          'normal': 'no',         'type': 'binary'},
        'cad':  {'name': 'Coronary Artery Disease', 'unit': '',             'range': None,          'normal': 'no',         'type': 'binary'},
        'appet':{'name': 'Appetite',                'unit': '',             'range': None,          'normal': 'good',       'type': 'binary'},
        'pe':   {'name': 'Pedal Edema',             'unit': '',             'range': None,          'normal': 'no',         'type': 'binary'},
        'ane':  {'name': 'Anemia',                  'unit': '',             'range': None,          'normal': 'no',         'type': 'binary'},
    }

    NUMERICAL_FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    CATEGORICAL_FEATURES = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    def __init__(self, use_knn_imputer: bool = True, knn_neighbors: int = 5):
        self.use_knn_imputer = use_knn_imputer
        self.knn_neighbors = knn_neighbors
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.feature_names = []
        self._fitted = False
        
        # Explicit mappings to prevent LabelEncoder sorting issues 
        self.cat_mappings = {
            'rbc':   {'normal': 0, 'abnormal': 1},
            'pc':    {'normal': 0, 'abnormal': 1},
            'pcc':   {'notpresent': 0, 'present': 1},
            'ba':    {'notpresent': 0, 'present': 1},
            'htn':   {'no': 0, 'yes': 1},
            'dm':    {'no': 0, 'yes': 1},
            'cad':   {'no': 0, 'yes': 1},
            'appet': {'good': 0, 'poor': 1},
            'pe':    {'no': 0, 'yes': 1},
            'ane':   {'no': 0, 'yes': 1}
        }

    # ── Data Loading ──────────────────────────────────────────────────────
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the UCI CKD dataset from CSV or ARFF-converted CSV."""
        try:
            df = pd.read_csv(filepath, na_values=['?', '\t?', '', ' '])
        except Exception as e:
            raise ValueError(f"Error loading data from {filepath}: {e}")
        df.columns = df.columns.str.strip().str.lower()
        if 'classification' in df.columns:
            df.rename(columns={'classification': 'class'}, inplace=True)
        return df

    # ── Cleaning ──────────────────────────────────────────────────────────
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace, normalise categorical values, fix tab chars."""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (df[col].astype(str)
                           .str.replace('\t', '', regex=False)
                           .str.strip()
                           .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'none': np.nan}))

        # Normalise target
        if 'class' in df.columns:
            df['class'] = df['class'].replace({'ckdt': 'ckd'})

        # Normalise binary columns
        for col in ['htn', 'dm', 'cad', 'pe', 'ane']:
            if col in df.columns:
                df[col] = df[col].replace({'\tyes': 'yes', '\tno': 'no', ' yes': 'yes', ' no': 'no'})

        if 'appet' in df.columns:
            df['appet'] = df['appet'].replace({'\tgood': 'good', '\tpoor': 'poor'})

        return df

    def convert_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # ── Validation ────────────────────────────────────────────────────────
    def validate_input(self, patient_data: Dict[str, Any]) -> Dict[str, list]:
        """
        Validate patient input values against medical ranges.
        Returns dict with 'warnings' and 'errors' lists.
        """
        result = {'warnings': [], 'errors': []}
        for feature, value in patient_data.items():
            info = self.FEATURE_INFO.get(feature)
            if not info or info['type'] in ('categorical', 'binary'):
                continue
            med_range = info.get('range')
            if med_range and value is not None:
                low, high = med_range
                if value < low or value > high:
                    result['errors'].append(
                        f"{info['name']}: {value} {info['unit']} is outside the medically plausible range ({low}–{high})"
                    )
            normal = info.get('normal')
            if normal and isinstance(normal, tuple) and value is not None:
                nlow, nhigh = normal
                if value < nlow:
                    result['warnings'].append(f"{info['name']}: {value} {info['unit']} is below normal range ({nlow}–{nhigh})")
                elif value > nhigh:
                    result['warnings'].append(f"{info['name']}: {value} {info['unit']} is above normal range ({nlow}–{nhigh})")
        return result

    def get_value_status(self, feature: str, value) -> str:
        """Return 'LOW', 'HIGH', 'NORMAL', or 'N/A' for a single value."""
        info = self.FEATURE_INFO.get(feature)
        if not info:
            return 'N/A'
        normal = info.get('normal')
        if normal is None:
            return 'N/A'
        if isinstance(normal, tuple):
            nlow, nhigh = normal
            try:
                v = float(value)
            except (TypeError, ValueError):
                return 'N/A'
            if v < nlow:
                return 'LOW'
            elif v > nhigh:
                return 'HIGH'
            return 'NORMAL'
        if isinstance(normal, str):
            return 'NORMAL' if str(value).lower() == normal.lower() else 'ABNORMAL'
        return 'N/A'

    # ── eGFR (CKD-EPI 2021) ──────────────────────────────────────────────
    @staticmethod
    def calculate_egfr(creatinine: float, age: float, is_female: bool = False) -> float:
        """
        Calculate eGFR using the CKD-EPI 2021 race-free equation.

        eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(-1.200) × 0.9938^Age [× 1.012 if female]

        Where κ = 0.7 (F) / 0.9 (M), α = -0.241 (F) / -0.302 (M)
        """
        if pd.isna(creatinine) or pd.isna(age) or creatinine <= 0 or age <= 0:
            return np.nan

        if is_female:
            kappa, alpha = 0.7, -0.241
        else:
            kappa, alpha = 0.9, -0.302

        scr_ratio = creatinine / kappa
        egfr = 142 * (min(scr_ratio, 1.0) ** alpha) * (max(scr_ratio, 1.0) ** -1.200) * (0.9938 ** age)
        if is_female:
            egfr *= 1.012

        return round(egfr, 2)

    @staticmethod
    def classify_ckd_stage(egfr: float) -> int:
        """
        KDIGO CKD staging from eGFR:
        Stage 1: ≥ 90  |  Stage 2: 60–89  |  Stage 3: 30–59  |  Stage 4: 15–29  |  Stage 5: < 15
        """
        if pd.isna(egfr):
            return 0
        if egfr >= 90:
            return 1
        if egfr >= 60:
            return 2
        if egfr >= 30:
            return 3
        if egfr >= 15:
            return 4
        return 5

    @staticmethod
    def get_stage_description(stage: int) -> str:
        descriptions = {
            0: "Unknown",
            1: "Normal or High (>=90 mL/min) - kidney damage with normal function",
            2: "Mildly Decreased (60-89 mL/min)",
            3: "Moderately Decreased (30-59 mL/min)",
            4: "Severely Decreased (15-29 mL/min)",
            5: "Kidney Failure (<15 mL/min)",
        }
        return descriptions.get(stage, "Unknown")

    # ── Feature Engineering ───────────────────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['egfr'] = df.apply(
            lambda row: self.calculate_egfr(row.get('sc', np.nan), row.get('age', np.nan)),
            axis=1
        )
        df['ckd_stage'] = df['egfr'].apply(self.classify_ckd_stage)
        return df

    # ── Missing Value Handling ────────────────────────────────────────────
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
        df = df.copy()
        imp_info = {}

        # Numerical — KNN or median
        num_cols = [c for c in self.NUMERICAL_FEATURES if c in df.columns]
        if num_cols:
            if fit:
                if getattr(self, 'use_knn_imputer', True):
                    self.numerical_imputer = KNNImputer(n_neighbors=self.knn_neighbors, weights='distance')
                else:
                    self.numerical_imputer = SimpleImputer(strategy='median')
                df[num_cols] = self.numerical_imputer.fit_transform(df[num_cols])
            elif self.numerical_imputer is not None:
                df[num_cols] = self.numerical_imputer.transform(df[num_cols])
            imp_info['numerical'] = 'KNN' if getattr(self, 'use_knn_imputer', True) else 'median'

        # Categorical — most frequent
        cat_cols = [c for c in self.CATEGORICAL_FEATURES if c in df.columns]
        if cat_cols:
            if fit:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = self.categorical_imputer.fit_transform(df[cat_cols])
            elif self.categorical_imputer is not None:
                df[cat_cols] = self.categorical_imputer.transform(df[cat_cols])
            imp_info['categorical'] = 'most_frequent'

        # Engineered features
        if 'egfr' in df.columns:
            med = df['egfr'].median()
            df['egfr'] = df['egfr'].fillna(med if not pd.isna(med) else 90.0)
        if 'ckd_stage' in df.columns:
            mode = df['ckd_stage'].mode()
            df['ckd_stage'] = df['ckd_stage'].fillna(mode.iloc[0] if len(mode) > 0 else 1)

        df = df.fillna(0)
        return df, imp_info

    # ── Encoding ──────────────────────────────────────────────────────────
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        for col in self.CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            mapping = self.cat_mappings.get(col, {})
            # Map known values to integers, unknown/NaN to -1
            df[col] = df[col].astype(str).str.lower().map(mapping).fillna(-1).astype(int)
            # if fit is True and imputation hasn't happened yet, NaNs might be present.
            # But imputation happens before encoding in fit_transform, so -1s should be rare.
        return df

    def encode_target(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
        df = df.copy()
        enc_map = {'notckd': 0, 'ckd': 1}
        if 'class' in df.columns:
            df['class'] = df['class'].astype(str).str.lower().map(enc_map).fillna(0).astype(int)
        return df, enc_map

    # ── Pipeline ──────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        info = {}
        df = self.clean_data(df)
        df = self.convert_numerical_columns(df)
        df = self.engineer_features(df)
        df, imp_info = self.handle_missing_values(df, fit=True)
        info['imputation'] = imp_info
        df = self.encode_categorical(df, fit=True)
        df, target_map = self.encode_target(df, fit=True)
        info['target_encoding'] = target_map
        self.feature_names = [c for c in df.columns if c not in ['class', 'ckd_stage']]
        self._fitted = True
        return df, info

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Backward compatibility for models saved before '_fitted' was added
        if not getattr(self, '_fitted', True):
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")
        df = self.clean_data(df)
        df = self.convert_numerical_columns(df)
        df = self.engineer_features(df)
        df, _ = self.handle_missing_values(df, fit=False)
        df = self.encode_categorical(df, fit=False)
        # Don't encode target for new patient data (no 'class' column)
        if 'class' in df.columns:
            df, _ = self.encode_target(df, fit=False)
        return df

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'CKDDataPreprocessor':
        obj = joblib.load(filepath)
        if not isinstance(obj, CKDDataPreprocessor):
            raise TypeError(f"Expected CKDDataPreprocessor, got {type(obj)}")
        return obj


def create_sample_dataset(n_samples: int = 400, random_state: int = 42) -> pd.DataFrame:
    """
    Create a clinically-correlated synthetic CKD dataset.
    
    Unlike purely random generation, this creates realistic biomarker 
    correlations: CKD patients get elevated creatinine/urea, low hemoglobin,
    abnormal urinalysis, and comorbidities. Non-CKD patients get normal values.
    """
    rng = np.random.RandomState(random_state)
    n_ckd = int(n_samples * 0.625)
    n_healthy = n_samples - n_ckd

    def _gen_group(n, ckd: bool):
        if ckd:
            return {
                'age':   rng.randint(40, 80, n),
                'bp':    rng.randint(90, 180, n),
                'sg':    rng.choice([1.005, 1.010, 1.015], n, p=[0.4, 0.4, 0.2]),
                'al':    rng.choice([0, 1, 2, 3, 4], n, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
                'su':    rng.choice([0, 1, 2, 3, 4], n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                'rbc':   rng.choice(['normal', 'abnormal'], n, p=[0.3, 0.7]),
                'pc':    rng.choice(['normal', 'abnormal'], n, p=[0.3, 0.7]),
                'pcc':   rng.choice(['notpresent', 'present'], n, p=[0.4, 0.6]),
                'ba':    rng.choice(['notpresent', 'present'], n, p=[0.5, 0.5]),
                'bgr':   rng.randint(120, 400, n),
                'bu':    np.clip(rng.normal(80, 30, n), 20, 300).astype(int),
                'sc':    np.clip(rng.lognormal(0.7, 0.6, n), 1.2, 15).round(2),
                'sod':   rng.randint(120, 140, n),
                'pot':   np.clip(rng.normal(5.5, 1.0, n), 3.0, 9.0).round(1),
                'hemo':  np.clip(rng.normal(9.5, 2.0, n), 4.0, 14.0).round(1),
                'pcv':   rng.randint(20, 40, n),
                'wc':    rng.randint(6000, 18000, n),
                'rc':    np.clip(rng.normal(3.8, 0.8, n), 2.5, 5.5).round(1),
                'htn':   rng.choice(['yes', 'no'], n, p=[0.7, 0.3]),
                'dm':    rng.choice(['yes', 'no'], n, p=[0.6, 0.4]),
                'cad':   rng.choice(['yes', 'no'], n, p=[0.3, 0.7]),
                'appet': rng.choice(['good', 'poor'], n, p=[0.3, 0.7]),
                'pe':    rng.choice(['yes', 'no'], n, p=[0.5, 0.5]),
                'ane':   rng.choice(['yes', 'no'], n, p=[0.6, 0.4]),
                'class': ['ckd'] * n,
            }
        else:
            return {
                'age':   rng.randint(18, 70, n),
                'bp':    rng.randint(60, 130, n),
                'sg':    rng.choice([1.015, 1.020, 1.025], n, p=[0.2, 0.4, 0.4]),
                'al':    rng.choice([0, 1], n, p=[0.9, 0.1]),
                'su':    rng.choice([0, 1], n, p=[0.9, 0.1]),
                'rbc':   rng.choice(['normal', 'abnormal'], n, p=[0.9, 0.1]),
                'pc':    rng.choice(['normal', 'abnormal'], n, p=[0.9, 0.1]),
                'pcc':   rng.choice(['notpresent', 'present'], n, p=[0.95, 0.05]),
                'ba':    rng.choice(['notpresent', 'present'], n, p=[0.95, 0.05]),
                'bgr':   rng.randint(70, 150, n),
                'bu':    np.clip(rng.normal(15, 5, n), 5, 40).astype(int),
                'sc':    np.clip(rng.normal(0.9, 0.2, n), 0.4, 1.4).round(2),
                'sod':   rng.randint(135, 148, n),
                'pot':   np.clip(rng.normal(4.2, 0.4, n), 3.2, 5.2).round(1),
                'hemo':  np.clip(rng.normal(14.5, 1.5, n), 11.0, 18.0).round(1),
                'pcv':   rng.randint(36, 52, n),
                'wc':    rng.randint(4000, 11000, n),
                'rc':    np.clip(rng.normal(5.0, 0.5, n), 3.8, 6.5).round(1),
                'htn':   rng.choice(['yes', 'no'], n, p=[0.15, 0.85]),
                'dm':    rng.choice(['yes', 'no'], n, p=[0.1, 0.9]),
                'cad':   rng.choice(['yes', 'no'], n, p=[0.05, 0.95]),
                'appet': rng.choice(['good', 'poor'], n, p=[0.9, 0.1]),
                'pe':    rng.choice(['yes', 'no'], n, p=[0.05, 0.95]),
                'ane':   rng.choice(['yes', 'no'], n, p=[0.08, 0.92]),
                'class': ['notckd'] * n,
            }

    ckd_data = _gen_group(n_ckd, ckd=True)
    healthy_data = _gen_group(n_healthy, ckd=False)

    df = pd.concat([pd.DataFrame(ckd_data), pd.DataFrame(healthy_data)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Sprinkle ~8% missing values (not in 'class')
    for col in df.columns:
        if col != 'class':
            mask = rng.random(len(df)) < 0.08
            df.loc[mask, col] = np.nan

    return df


if __name__ == '__main__':
    pre = CKDDataPreprocessor()
    df = create_sample_dataset()
    print("Dataset shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    df_p, info = pre.fit_transform(df)
    print("Processed shape:", df_p.shape)
    print("eGFR range:", df_p['egfr'].min(), "–", df_p['egfr'].max())
    print("Stages:\n", df_p['ckd_stage'].value_counts().sort_index())