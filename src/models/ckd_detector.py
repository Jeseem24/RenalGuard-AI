"""
RenalGuard AI - CKD Detection Model
Binary Classification: CKD vs Not CKD
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class CKDDetector:
    """
    Binary classifier for CKD detection
    Trains and compares multiple models:
    - XGBoost
    - LightGBM  
    - Random Forest
    - Logistic Regression
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.metrics = {}
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize all models with default parameters"""
        self.models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train all models with cross-validation and select the best one
        
        Args:
            X: Feature dataframe
            y: Target series (binary: 0/1)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with metrics for all models
        """
        self.feature_names = list(X.columns)
        results = {}
        
        # Handle class imbalance - calculate scale_pos_weight
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        print(f"Training samples: {len(X)}")
        print(f"Class distribution - Not CKD: {neg_count}, CKD: {pos_count}")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        print("-" * 50)
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Update class weights for imbalanced data
            if name == 'XGBoost':
                model.set_params(scale_pos_weight=scale_pos_weight)
            elif name == 'LightGBM':
                model.set_params(class_weight='balanced')
            elif name == 'RandomForest':
                model.set_params(class_weight='balanced')
            elif name == 'LogisticRegression':
                model.set_params(class_weight='balanced')
            
            # Cross-validation predictions
            y_pred = cross_val_predict(model, X, y, cv=skf, method='predict')
            y_pred_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y, y_pred, y_pred_proba)
            results[name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Select best model based on ROC-AUC (you can change this criterion)
        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        # Fit best model on full data
        self.best_model.fit(X, y)
        self.metrics = results[best_model_name]
        
        print("\n" + "=" * 50)
        print(f"Best Model: {best_model_name}")
        print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        print("=" * 50)
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Calculate all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'sensitivity': recall_score(y_true, y_pred, average='binary'),  # Same as recall
            'specificity': recall_score(y_true, y_pred, pos_label=0, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Calculate precision-recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.best_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.best_model.predict_proba(X)
    
    def get_risk_score(self, X: pd.DataFrame) -> float:
        """Get risk score (probability of CKD) as percentage"""
        proba = self.predict_proba(X)
        return float(proba[0, 1] * 100)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_[0])
        else:
            return pd.DataFrame()
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def save(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        print(f"Model loaded from {filepath}")
    
    def get_summary(self) -> str:
        """Get a summary string of the model"""
        if self.best_model is None:
            return "Model not trained yet."
        
        summary = f"""
        CKD Detection Model Summary
        ===========================
        Best Model: {self.best_model_name}
        
        Performance Metrics:
        - Accuracy: {self.metrics['accuracy']:.4f}
        - Precision: {self.metrics['precision']:.4f}
        - Recall (Sensitivity): {self.metrics['recall']:.4f}
        - Specificity: {self.metrics['specificity']:.4f}
        - F1 Score: {self.metrics['f1']:.4f}
        - ROC-AUC: {self.metrics['roc_auc']:.4f}
        - PR-AUC: {self.metrics['pr_auc']:.4f}
        
        Features Used: {len(self.feature_names)}
        """
        return summary


class CKDStageClassifier:
    """
    Multi-class classifier for CKD Stage (1-5)
    Only applies to patients predicted as CKD positive
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self.stage_offset = 1  # Stages are 1-5, model needs 0-4
        
    def train(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the multi-class stage classifier
        
        Args:
            X: Feature dataframe
            y: Target series (stage: 1-5)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with metrics
        """
        self.feature_names = list(X.columns)
        
        # Filter out stage 0 (unknown) samples
        valid_mask = y > 0
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Convert stages 1-5 to 0-4 for XGBoost compatibility
        y_encoded = y - self.stage_offset
        
        print(f"Training Stage Classifier with {len(X)} samples")
        print(f"Stage distribution:")
        print(y.value_counts().sort_index())
        
        # Initialize XGBoost for multi-class
        num_classes = len(y.unique())
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train the model with encoded labels
        self.model.fit(X, y_encoded)
        
        # Cross-validation predictions
        skf = StratifiedKFold(n_splits=min(cv_folds, 3), shuffle=True, random_state=self.random_state)
        y_pred_encoded = cross_val_predict(self.model, X, y_encoded, cv=skf)
        
        # Convert back to original scale
        y_pred = y_pred_encoded + self.stage_offset
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'macro_f1': f1_score(y, y_pred, average='macro'),
            'weighted_f1': f1_score(y, y_pred, average='weighted'),
            'classification_report': classification_report(y, y_pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        print(f"\nStage Classifier Metrics:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  Macro F1: {self.metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {self.metrics['weighted_f1']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CKD stage"""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get stage prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']


if __name__ == '__main__':
    # Test the models
    from preprocessing.data_preprocessor import create_sample_dataset, CKDDataPreprocessor
    
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Preprocess
    preprocessor = CKDDataPreprocessor()
    df_processed, _ = preprocessor.fit_transform(df)
    
    # Prepare features and target
    feature_cols = [col for col in df_processed.columns if col not in ['class', 'ckd_stage']]
    X = df_processed[feature_cols]
    y = df_processed['class']
    
    # Train CKD detector
    print("\n" + "=" * 50)
    print("Training CKD Detection Model")
    print("=" * 50)
    
    detector = CKDDetector()
    results = detector.train(X, y)
    
    print("\n" + detector.get_summary())
    
    # Train Stage Classifier
    print("\n" + "=" * 50)
    print("Training CKD Stage Classifier")
    print("=" * 50)
    
    stage_classifier = CKDStageClassifier()
    stage_results = stage_classifier.train(X, df_processed['ckd_stage'])
    
    print("\nModels tested successfully!")
