"""
RenalGuard AI - Model Training Script
"""
# ─── PATH SETUP ──────────────────────────────────────────────────────────────
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
from src.models.ckd_detector import CKDDetector, CKDStageClassifier


def main():
    print("=" * 60)
    print("  RenalGuard AI — Model Training Pipeline")
    print("=" * 60)

    models_dir = Path(__file__).parent.parent / 'reports' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    print("\n📊 Step 1: Generating clinically-correlated synthetic dataset...")
    df = create_sample_dataset(n_samples=400)
    print(f"   Shape: {df.shape}")
    print(f"   CKD: {(df['class'] == 'ckd').sum()} | Not CKD: {(df['class'] == 'notckd').sum()}")

    # 2. Preprocessing
    print("\n🔧 Step 2: Preprocessing (KNN imputation, encoding, eGFR)...")
    pre = CKDDataPreprocessor(use_knn_imputer=True)
    df_p, info = pre.fit_transform(df)
    feature_cols = [c for c in df_p.columns if c not in ['class', 'ckd_stage']]
    X = df_p[feature_cols]
    y = df_p['class']
    y_stage = df_p['ckd_stage']
    print(f"   Features: {len(feature_cols)}")
    print(f"   eGFR range: {df_p['egfr'].min():.1f} – {df_p['egfr'].max():.1f}")
    print(f"   Stage distribution:\n{y_stage.value_counts().sort_index().to_string()}")

    # 3. Binary detector
    print("\n🤖 Step 3: Training CKD Binary Detector (incl. stacking ensemble)...")
    print("-" * 60)
    det = CKDDetector()
    results = det.train(X, y, cv_folds=5)
    det.save(models_dir / 'ckd_detector.joblib')

    # 4. Stage classifier
    print("\n🔢 Step 4: Training CKD Stage Classifier...")
    print("-" * 60)
    stg = CKDStageClassifier()
    stg.train(X, y_stage, cv_folds=3)
    stg.save(models_dir / 'stage_classifier.joblib')

    # 5. Save preprocessor
    print("\n💾 Step 5: Saving preprocessor...")
    pre.save(models_dir / 'preprocessor.joblib')

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  🏆 Best Model: {det.best_model_name}")
    print(f"     ROC-AUC:   {det.metrics.get('roc_auc', 0):.4f}")
    print(f"     F1:        {det.metrics.get('f1', 0):.4f}")
    print(f"\n  Saved to: {models_dir}")
    print("\n  🚀 Launch: streamlit run app/main.py")


if __name__ == '__main__':
    main()