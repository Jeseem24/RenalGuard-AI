"""
RenalGuard AI - Training Script
Run this script to train and save models for the first time
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_preprocessor import CKDDataPreprocessor, create_sample_dataset
from src.models.ckd_detector import CKDDetector, CKDStageClassifier
import joblib
import os

def main():
    print("=" * 60)
    print("RenalGuard AI - Model Training Script")
    print("=" * 60)
    
    # Create directories
    models_dir = Path(__file__).parent.parent / 'reports' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create/Load Dataset
    print("\n📊 Step 1: Loading Dataset...")
    df = create_sample_dataset()
    print(f"   Dataset shape: {df.shape}")
    print(f"   CKD cases: {(df['class'] == 'ckd').sum()}")
    print(f"   Non-CKD cases: {(df['class'] == 'notckd').sum()}")
    
    # Step 2: Preprocess Data
    print("\n🔧 Step 2: Preprocessing Data...")
    preprocessor = CKDDataPreprocessor()
    df_processed, info = preprocessor.fit_transform(df)
    print(f"   Processed shape: {df_processed.shape}")
    
    # Prepare features and targets
    feature_cols = [col for col in df_processed.columns if col not in ['class', 'ckd_stage']]
    X = df_processed[feature_cols]
    y = df_processed['class']
    y_stage = df_processed['ckd_stage']
    
    print(f"   Features: {len(feature_cols)}")
    
    # Step 3: Train CKD Detector
    print("\n🤖 Step 3: Training CKD Detection Model...")
    print("-" * 50)
    detector = CKDDetector()
    results = detector.train(X, y, cv_folds=5)
    
    # Print results
    print("\n📈 Model Comparison Results:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Save detector
    detector_path = models_dir / 'ckd_detector.joblib'
    detector.save(detector_path)
    
    # Step 4: Train Stage Classifier
    print("\n🔢 Step 4: Training CKD Stage Classifier...")
    print("-" * 50)
    stage_clf = CKDStageClassifier()
    stage_results = stage_clf.train(X, y_stage, cv_folds=3)
    
    print(f"\nStage Classifier Metrics:")
    print(f"  Accuracy:    {stage_results['accuracy']:.4f}")
    print(f"  Macro F1:    {stage_results['macro_f1']:.4f}")
    print(f"  Weighted F1: {stage_results['weighted_f1']:.4f}")
    
    # Save stage classifier
    stage_path = models_dir / 'stage_classifier.joblib'
    stage_clf.save(stage_path)
    
    # Step 5: Save Preprocessor
    print("\n💾 Step 5: Saving Preprocessor...")
    preprocessor_path = models_dir / 'preprocessor.joblib'
    preprocessor.save(preprocessor_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nSaved files:")
    print(f"  - {detector_path}")
    print(f"  - {stage_path}")
    print(f"  - {preprocessor_path}")
    
    print(f"\n🏆 Best Model: {detector.best_model_name}")
    print(f"   ROC-AUC: {detector.metrics['roc_auc']:.4f}")
    
    print("\n🚀 You can now run the Streamlit app:")
    print("   streamlit run app/main.py")
    
    return detector, stage_clf, preprocessor


if __name__ == '__main__':
    main()
