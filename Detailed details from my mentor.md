================================================================
PROJECT BRIEF: RENALGUARD AI
COMPLETE SPECIFICATION DOCUMENT
FOR AI AGENT TO BUILD THE ENTIRE PROJECT
================================================================

TABLE OF CONTENTS:

1.  PROJECT CONTEXT & HACKATHON REQUIREMENTS
2.  PROBLEM STATEMENT
3.  SOLUTION OVERVIEW
4.  COMPLETE TECH STACK
5.  DATASET DETAILS
6.  PROJECT FOLDER STRUCTURE
7.  DATA PREPROCESSING PIPELINE
8.  MODEL 1: CKD DETECTION (Binary Classification)
9.  MODEL 2: CKD STAGE CLASSIFICATION (Multi-class)
10. SHAP EXPLAINABILITY INTEGRATION
11. LLM CONVERSATIONAL INTERFACE
12. PDF CLINICAL REPORT GENERATION
13. STREAMLIT FRONTEND (Complete UI Specification)
14. BACKEND API (FastAPI - Optional)
15. GITHUB REPOSITORY REQUIREMENTS
16. README.md SPECIFICATION
17. PPT CONTENT (Slide-by-Slide)
18. DEMO VIDEO SCRIPT
19. DEPLOYMENT INSTRUCTIONS
20. MEDICAL DOMAIN KNOWLEDGE
21. EVALUATION CRITERIA ALIGNMENT
22. FILE-BY-FILE CODE SPECIFICATIONS
23. TESTING REQUIREMENTS
24. FINAL CHECKLIST

================================================================
SECTION 1: PROJECT CONTEXT & HACKATHON REQUIREMENTS
================================================================

HACKATHON NAME: AI4Dev '26
FULL TITLE: "AI-Enabled Transformative Technologies for Global
Development"
ORGANIZED BY: PSG College of Technology, Coimbatore (Coding Club)
LEVEL: National-Level (Pan-India)
EVENT DATE: March 28, 2026
SUBMISSION DEADLINE: March 8, 2026
MODE: Round 1 = Online submission, Round 2 = Offline at PSG Tech

SELECTED DOMAIN: Healthcare and Life Sciences
DOMAIN DESCRIPTION: "Innovative AI applications in medical
diagnosis, drug discovery, and healthcare
delivery"

TEAM SIZE: 2-3 members
TRL REQUIREMENT: TRL Level 3 or above (Experimental Proof of
Concept)

WHAT TRL 3 MEANS:

- Analytical and experimental critical function and/or
  characteristic proof of concept
- Basically: A working prototype that demonstrates the core
  functionality, not a production-ready system

ROUND 1 SUBMISSION REQUIREMENTS:

1. Working Proof of Concept (PoC) with TRL Level 3+
2. Project PPT explaining the solution (template provided
   by hackathon)
3. GitHub repository link with code
4. Demo video URL showcasing the PoC

ROUND 1 EVALUATION CRITERIA:

- Project PPT quality and clarity
- GitHub code quality and documentation
- Demo video clarity and presentation

ROUND 2 EVALUATION CRITERIA (if shortlisted):

- Presentation and creativity
- Solution viability, real world issues and address technique
- Originality of the solution

JURY MEMBERS (Important - solution should impress them):

1. Mr. Madhusudhan - Senior Manager, Product Management,
   AI SoC at Intel, Bangalore
   → Cares about: Edge AI, optimization, inference efficiency
2. Dr. C. S. Saravana Kumar - Senior Software Architect
   at Bosch Global Software Technologies
   → Cares about: Clean architecture, code quality, scalability
3. Dr. Srinivasan Aruchamy - Principal Scientist at CSIR-CMERI
   → Cares about: Scientific rigor, research depth, real-world
   impact
4. Mr. Asadh Sheriff - Director/Software Architect at Okta,
   Bangalore
   → Cares about: System design, security, UX
5. Ms. Usha Rengaraju - Kaggle Grandmaster, Head of Research
   at Exa Protocol
   → Cares about: ML rigor, novel approaches, proper evaluation,
   explainability

PRIZE POOL: ₹60,000 (1st: ₹30,000, 2nd: ₹20,000,
3rd: ₹10,000)

================================================================
SECTION 2: PROBLEM STATEMENT
================================================================

TITLE: The Silent Killer - Kidney Disease Detection Crisis

PROBLEM:

- 850 million people worldwide have Chronic Kidney Disease (CKD)
- 90% of cases are diagnosed TOO LATE - when kidneys are
  already severely damaged
- By the time symptoms appear, only 15-30% kidney function
  remains
- Treatment at late stages: Dialysis (₹15,000-25,000/month)
  or Transplant (₹5-15 lakhs)

WHY THIS HAPPENS:

- Kidney disease shows NO SYMPTOMS in early stages (Stage 1-3)
- Early detection tests are expensive or unavailable in
  rural areas
- Doctors often catch it only at Stage 4 or 5 (out of 5 stages)
- No simple, affordable tool exists to predict kidney
  disease risk early
- India has the 2nd highest CKD burden globally

WHO SUFFERS MOST:

- People in developing countries (80% of CKD deaths)
- Rural populations with limited healthcare access
- Diabetic and hypertensive patients (highest risk groups)
- People who can't afford expensive treatments

CKD STAGES (Medical Knowledge):

- Stage 1: eGFR ≥ 90 (Kidney damage with normal function)
- Stage 2: eGFR 60-89 (Mild decrease in function)
- Stage 3a: eGFR 45-59 (Mild to moderate decrease)
- Stage 3b: eGFR 30-44 (Moderate to severe decrease)
- Stage 4: eGFR 15-29 (Severe decrease)
- Stage 5: eGFR < 15 (Kidney failure)

eGFR = estimated Glomerular Filtration Rate (key kidney
function metric, measured in mL/min/1.73m²)

KDIGO GUIDELINES (Clinical Reference):

- KDIGO = Kidney Disease: Improving Global Outcomes
- International clinical practice guidelines for CKD
- Our recommendations in the app should align with these
  guidelines

================================================================
SECTION 3: SOLUTION OVERVIEW
================================================================

PROJECT NAME: RenalGuard AI
TAGLINE: "Early Detection & Prediction Platform for Chronic
Kidney Disease"

ONE-LINE DESCRIPTION:
An AI-powered clinical decision support tool that predicts
kidney disease risk EARLY using basic blood/urine test results
available even in small clinics, with explainable AI insights
and LLM-powered conversational assistance.

KEY FEATURES (In priority order):

MUST BUILD (Priority 1 - Core):

1. CKD Detection Model - Binary classification (CKD / Not CKD)
2. CKD Stage Classification - Multi-class (Stage 1-5)
3. SHAP Explainability - Visual explanations of WHY the AI
   made its prediction
4. Streamlit Web Application - Interactive UI for input
   and results
5. Clean GitHub Repository - Well-structured, documented code

SHOULD BUILD (Priority 2 - Differentiators): 6. LLM Conversational Interface - Gemini API powered chat
that explains results 7. PDF Clinical Report Generation - Auto-generated medical
screening report 8. Data Visualization Dashboard - EDA plots, risk distribution

MENTION IN PPT ONLY (Priority 3 - Future Scope): 9. ABDM (Ayushman Bharat Digital Mission) integration concept 10. ONNX/TFLite edge deployment 11. Offline mode capability 12. Multi-language support 13. Progression prediction over time 14. Mobile app version

HOW IT WORKS (User Flow):

1. Doctor/Health worker opens the web app
2. Enters patient's basic test values (creatinine, blood urea,
   hemoglobin, etc.)
3. AI predicts: CKD risk (yes/no) + Stage + Risk Score
4. SHAP shows: WHY the prediction was made (which values
   contributed most)
5. LLM explains: Results in simple language with recommendations
6. PDF report: Can be downloaded and given to patient/doctor
7. Chat: Doctor can ask follow-up questions

================================================================
SECTION 4: COMPLETE TECH STACK
================================================================

PROGRAMMING LANGUAGE: Python 3.9+

ML/AI LIBRARIES:

- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (preprocessing, metrics, model selection)
- xgboost (primary ML model)
- lightgbm (secondary ML model for comparison)
- shap (explainable AI - CRITICAL)
- matplotlib (plotting)
- seaborn (statistical visualization)
- plotly (interactive plots for Streamlit)
- joblib (model serialization)
- imbalanced-learn (SMOTE for class imbalance, if needed)

LLM INTEGRATION:

- google-generativeai (Gemini API - free tier)
  OR
- groq (Groq API - free, fast inference)
  OR
- openai (if user has API key)

FRONTEND:

- streamlit (primary UI framework)
- streamlit-extras (additional components)
- streamlit-lottie (animations, optional)
- streamlit-option-menu (navigation)

PDF GENERATION:

- reportlab (PDF creation)
  OR
- fpdf2 (simpler alternative)

DEPLOYMENT:

- Streamlit Cloud (free, easiest)
  OR
- Hugging Face Spaces (free)
  OR
- Railway.app (free tier)

VERSION CONTROL:

- Git + GitHub

OTHER:

- python-dotenv (environment variables for API keys)
- Pillow (image handling)

REQUIREMENTS.TXT:
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.43.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
streamlit>=1.28.0
google-generativeai>=0.3.0
reportlab>=4.0.0
joblib>=1.3.0
imbalanced-learn>=0.11.0
python-dotenv>=1.0.0
fpdf2>=2.7.0
Pillow>=10.0.0

================================================================
SECTION 5: DATASET DETAILS
================================================================

PRIMARY DATASET: UCI Chronic Kidney Disease Dataset

SOURCE:

- Kaggle: https://www.kaggle.com/datasets/mansoordaku/ckdisease
- UCI ML Repository: https://archive.ics.uci.edu/dataset/336/
  chronic+kidney+disease

DATASET CHARACTERISTICS:

- Total samples: 400
- Features: 24 + 1 target
- Target variable: "classification" (ckd / notckd)
- Missing values: YES (significant - needs handling)
- Class distribution: ~250 CKD, ~150 Not CKD (imbalanced)

ALL 24 FEATURES WITH DESCRIPTIONS:

| Column         | Full Name               | Type      | Unit         | Normal Range       |
| -------------- | ----------------------- | --------- | ------------ | ------------------ |
| age            | Age                     | Numerical | years        | -                  |
| bp             | Blood Pressure          | Numerical | mm/Hg        | 80-120             |
| sg             | Specific Gravity        | Nominal   | -            | 1.005-1.025        |
| al             | Albumin                 | Nominal   | -            | 0-5                |
| su             | Sugar                   | Nominal   | -            | 0-5                |
| rbc            | Red Blood Cells         | Nominal   | -            | normal/abnormal    |
| pc             | Pus Cell                | Nominal   | -            | normal/abnormal    |
| pcc            | Pus Cell Clumps         | Nominal   | -            | present/notpresent |
| ba             | Bacteria                | Nominal   | -            | present/notpresent |
| bgr            | Blood Glucose Random    | Numerical | mgs/dl       | 70-140             |
| bu             | Blood Urea              | Numerical | mgs/dl       | 7-20               |
| sc             | Serum Creatinine        | Numerical | mgs/dl       | 0.6-1.2            |
| sod            | Sodium                  | Numerical | mEq/L        | 136-145            |
| pot            | Potassium               | Numerical | mEq/L        | 3.5-5.0            |
| hemo           | Hemoglobin              | Numerical | gms          | 12-17              |
| pcv            | Packed Cell Volume      | Numerical | -            | 36-50              |
| wc             | White Blood Cell Count  | Numerical | cells/cumm   | 4500-11000         |
| rc             | Red Blood Cell Count    | Numerical | millions/cmm | 4.5-5.5            |
| htn            | Hypertension            | Nominal   | -            | yes/no             |
| dm             | Diabetes Mellitus       | Nominal   | -            | yes/no             |
| cad            | Coronary Artery Disease | Nominal   | -            | yes/no             |
| appet          | Appetite                | Nominal   | -            | good/poor          |
| pe             | Pedal Edema             | Nominal   | -            | yes/no             |
| ane            | Anemia                  | Nominal   | -            | yes/no             |
| classification | CKD or Not (TARGET)     | Nominal   | -            | ckd/notckd         |

DATA QUALITY ISSUES TO HANDLE:

1. Missing values in many columns (use median/mode imputation)
2. Some numerical columns stored as strings (need type conversion)
3. Tab characters (\t) in some values
4. "ckd\t" and "notckd" in target column (strip whitespace)
5. Class imbalance (handle with SMOTE or class weights)

FOR CKD STAGE CLASSIFICATION:

- The UCI dataset does NOT have stage labels
- We need to ENGINEER stage labels using eGFR calculation
- eGFR Formula (CKD-EPI or Cockcroft-Gault):

  Simplified approach using Serum Creatinine + Age + Gender:
  eGFR = 186 × (Serum Creatinine)^(-1.154) × (Age)^(-0.203)
  × (0.742 if female)

  Then map eGFR to stages:
  - eGFR >= 90 → Stage 1
  - eGFR 60-89 → Stage 2
  - eGFR 45-59 → Stage 3a
  - eGFR 30-44 → Stage 3b
  - eGFR 15-29 → Stage 4
  - eGFR < 15 → Stage 5

  Note: Since dataset doesn't have gender column explicitly,
  we can simplify or assume. This engineered label adds
  significant value to the project.

ALTERNATIVE/SUPPLEMENTARY DATA APPROACH:

- If agent wants more data, can use SMOTE to oversample
  minority classes
- Can create synthetic samples for underrepresented CKD stages
- Mention MIMIC-IV compatibility in documentation as
  future scope

================================================================
SECTION 6: PROJECT FOLDER STRUCTURE
================================================================

Create EXACTLY this folder structure:

RenalGuard-AI/
│
├── README.md
├── LICENSE (MIT License)
├── requirements.txt
├── .env.example
├── .gitignore
├── setup.py (optional)
│
├── app/
│ ├── **init**.py
│ ├── main.py ← Main Streamlit application
│ ├── pages/
│ │ ├── **init**.py
│ │ ├── home.py ← Landing page
│ │ ├── prediction.py ← CKD prediction page
│ │ ├── explainability.py ← SHAP visualizations page
│ │ ├── chat.py ← LLM chat interface page
│ │ ├── report.py ← PDF report page
│ │ ├── dashboard.py ← Data insights dashboard
│ │ └── about.py ← About the project
│ ├── components/
│ │ ├── **init**.py
│ │ ├── sidebar.py ← Sidebar navigation
│ │ ├── header.py ← Header component
│ │ ├── input_form.py ← Patient data input form
│ │ ├── result_card.py ← Prediction result display
│ │ └── shap_plots.py ← SHAP visualization components
│ ├── utils/
│ │ ├── **init**.py
│ │ ├── constants.py ← App constants, color schemes
│ │ ├── helpers.py ← Helper functions
│ │ └── validators.py ← Input validation
│ └── assets/
│ ├── logo.png ← RenalGuard AI logo
│ ├── style.css ← Custom CSS
│ └── animations/ ← Lottie animations (optional)
│
├── src/
│ ├── **init**.py
│ ├── data/
│ │ ├── **init**.py
│ │ ├── data_loader.py ← Load and validate dataset
│ │ ├── preprocessor.py ← Data cleaning & preprocessing
│ │ └── feature_engineer.py ← Feature engineering + eGFR
│ ├── models/
│ │ ├── **init**.py
│ │ ├── ckd_detector.py ← Binary CKD detection model
│ │ ├── stage_classifier.py ← CKD stage classification
│ │ ├── model_trainer.py ← Training pipeline
│ │ └── model_evaluator.py ← Evaluation metrics
│ ├── explainability/
│ │ ├── **init**.py
│ │ └── shap_explainer.py ← SHAP integration
│ ├── llm/
│ │ ├── **init**.py
│ │ ├── gemini_client.py ← Gemini API integration
│ │ └── prompts.py ← LLM prompt templates
│ └── reports/
│ ├── **init**.py
│ └── pdf_generator.py ← PDF report generation
│
├── data/
│ ├── raw/
│ │ └── kidney_disease.csv ← Original UCI dataset
│ ├── processed/
│ │ ├── cleaned_data.csv ← After preprocessing
│ │ └── featured_data.csv ← After feature engineering
│ └── README.md ← Data source attribution
│
├── models/
│ ├── ckd_detector.pkl ← Saved binary classifier
│ ├── stage_classifier.pkl ← Saved stage classifier
│ ├── scaler.pkl ← Saved feature scaler
│ ├── label_encoder.pkl ← Saved label encoders
│ └── model_metadata.json ← Model performance metrics
│
├── notebooks/
│ ├── 01_EDA_Analysis.ipynb
│ ├── 02_Data_Preprocessing.ipynb
│ ├── 03_Feature_Engineering.ipynb
│ ├── 04_Model_Training.ipynb
│ ├── 05_SHAP_Analysis.ipynb
│ └── 06_Model_Comparison.ipynb
│
├── reports/
│ └── sample_report.pdf ← Sample generated report
│
├── tests/
│ ├── **init**.py
│ ├── test_preprocessor.py
│ ├── test_models.py
│ └── test_explainer.py
│
├── docs/
│ ├── architecture.md
│ ├── model_card.md ← ML model documentation
│ └── screenshots/
│ ├── home.png
│ ├── prediction.png
│ ├── shap.png
│ └── report.png
│
└── .streamlit/
└── config.toml ← Streamlit theme configuration

================================================================
SECTION 7: DATA PREPROCESSING PIPELINE
================================================================

FILE: src/data/preprocessor.py

STEP-BY-STEP PREPROCESSING:

Step 1: Load Data

- Read kidney_disease.csv using pandas
- The dataset has 400 rows, 25 columns
- Encoding might be needed (try 'utf-8' or 'latin-1')

Step 2: Clean Column Names

- Strip whitespace from column names
- Make all column names lowercase
- Replace spaces with underscores

Step 3: Clean Target Variable

- Column "classification" has values: "ckd", "ckd\t",
  "notckd", "notckd\t"
- Strip all whitespace/tabs
- Map to binary: "ckd" → 1, "notckd" → 0

Step 4: Handle Data Type Issues

- Columns like 'pcv', 'wc', 'rc' may be stored as objects
  (strings)
- Convert to numeric using pd.to_numeric(errors='coerce')
- Some values have '\t' appended - strip before converting

Step 5: Handle Missing Values

- Numerical columns: Fill with MEDIAN (not mean, because of
  outliers)
- Categorical columns: Fill with MODE (most frequent value)
- Document the missing value percentages for each column

Step 6: Encode Categorical Variables

- 'rbc': normal→1, abnormal→0
- 'pc': normal→1, abnormal→0
- 'pcc': present→1, notpresent→0
- 'ba': present→1, notpresent→0
- 'htn': yes→1, no→0
- 'dm': yes→1, no→0
- 'cad': yes→1, no→0
- 'appet': good→1, poor→0
- 'pe': yes→1, no→0
- 'ane': yes→1, no→0
- Use LabelEncoder or manual mapping

Step 7: Feature Scaling

- Use StandardScaler on numerical features
- Save the scaler using joblib for later use
- Features to scale: age, bp, sg, bgr, bu, sc, sod, pot,
  hemo, pcv, wc, rc

Step 8: Feature Engineering (IMPORTANT)

- Calculate eGFR from serum creatinine and age:
  eGFR = 186 × (sc)^(-1.154) × (age)^(-0.203)
- Create CKD stage labels from eGFR:
  Stage 1: eGFR >= 90
  Stage 2: 60 <= eGFR < 90
  Stage 3a: 45 <= eGFR < 59
  Stage 3b: 30 <= eGFR < 44
  Stage 4: 15 <= eGFR < 29
  Stage 5: eGFR < 15
- Create risk categories: Low, Medium, High, Very High
- Create BUN/Creatinine ratio (bu / sc)
- Create Anemia indicator based on hemoglobin
  (< 12 for female, < 13 for male)

Step 9: Save Processed Data

- Save to data/processed/cleaned_data.csv
- Save to data/processed/featured_data.csv

Step 10: Split Data

- Train/Test split: 80/20
- Use stratified split (stratify=y) to maintain class balance
- Random state = 42 for reproducibility

================================================================
SECTION 8: MODEL 1 - CKD DETECTION (Binary Classification)
================================================================

FILE: src/models/ckd_detector.py

PURPOSE: Predict whether a patient has CKD or not (binary: 0/1)

MODELS TO TRAIN AND COMPARE:

1. XGBoost Classifier (PRIMARY - likely best performer)
2. LightGBM Classifier (SECONDARY - for comparison)
3. Random Forest Classifier (BASELINE - for comparison)
4. Logistic Regression (SIMPLE BASELINE)

FOR EACH MODEL:

Hyperparameters for XGBoost:
params = {
'n_estimators': 200,
'max_depth': 6,
'learning_rate': 0.1,
'subsample': 0.8,
'colsample_bytree': 0.8,
'min_child_weight': 3,
'gamma': 0.1,
'reg_alpha': 0.1,
'reg_lambda': 1.0,
'random_state': 42,
'eval_metric': 'logloss',
'use_label_encoder': False
}

TRAINING PROCEDURE:

1. Use Stratified K-Fold Cross Validation (k=5)
2. Track metrics for each fold
3. Use early stopping if applicable
4. Handle class imbalance using:
   - scale_pos_weight parameter in XGBoost
   - OR SMOTE oversampling
   - OR class_weight='balanced'

EVALUATION METRICS (MUST report all of these):

- Accuracy
- Precision (for both classes)
- Recall / Sensitivity (for both classes)
- F1-Score (for both classes)
- AUC-ROC Score
- AUC-PR Score (Precision-Recall curve)
- Confusion Matrix
- Classification Report
- Cross-validation mean and std

WHY THESE METRICS MATTER:

- In medical AI, RECALL for CKD class is MOST important
  (we don't want to miss actual CKD cases = false negatives)
- Precision matters too (false positives cause unnecessary
  anxiety)
- AUC-ROC shows overall discriminative ability
- Report sensitivity and specificity (medical terms for
  recall)

SAVE THE MODEL:

- Save best model using joblib: models/ckd_detector.pkl
- Save scaler: models/scaler.pkl
- Save metadata (metrics) as JSON: models/model_metadata.json

MODEL COMPARISON TABLE TO GENERATE:
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| XGBoost | ? | ? | ? | ? | ? |
| LightGBM | ? | ? | ? | ? | ? |
| Random Forest | ? | ? | ? | ? | ? |
| Logistic Reg | ? | ? | ? | ? | ? |

VISUALIZATION TO CREATE:

- ROC Curves (all models on one plot)
- Precision-Recall Curves
- Confusion Matrix heatmap (for best model)
- Feature Importance bar chart (for best model)
- Save all plots as PNG in docs/screenshots/

================================================================
SECTION 9: MODEL 2 - CKD STAGE CLASSIFICATION
================================================================

FILE: src/models/stage_classifier.py

PURPOSE: Classify the CKD stage (1-5 or 1, 2, 3a, 3b, 4, 5)

IMPORTANT NOTE:
This model only applies to patients PREDICTED as CKD positive.
It's a second-level classification.

APPROACH:

- Use the engineered eGFR-based stage labels from
  feature engineering
- This is a MULTI-CLASS classification problem
- Can simplify to 5 classes: Stage 1, 2, 3, 4, 5
  (merge 3a and 3b if too few samples)

MODEL: XGBoost with multi:softprob objective

Hyperparameters:
params = {
'objective': 'multi:softprob',
'num_class': 5,
'n_estimators': 200,
'max_depth': 5,
'learning_rate': 0.1,
'random_state': 42
}

EVALUATION METRICS:

- Multi-class accuracy
- Per-class precision, recall, F1
- Macro-averaged F1
- Weighted-averaged F1
- Confusion matrix (multi-class)
- Classification report

SAVE: models/stage_classifier.pkl

HANDLE LOW SAMPLE COUNTS:

- Some stages may have very few samples (especially Stage 1)
- Use SMOTE for multi-class or class weights
- If a stage has < 5 samples, consider merging with
  adjacent stage
- Document this decision clearly

================================================================
SECTION 10: SHAP EXPLAINABILITY INTEGRATION
================================================================

FILE: src/explainability/shap_explainer.py

THIS IS THE #1 DIFFERENTIATOR OF THE PROJECT.
BUILD THIS VERY WELL.

PURPOSE:
Explain WHY the AI made a specific prediction for each patient.
Show which features contributed most to the prediction.

IMPLEMENTATION:

import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
def **init**(self, model, X_train):
self.model = model
self.explainer = shap.TreeExplainer(model)
self.X_train = X_train

    def get_shap_values(self, patient_data):
        """Get SHAP values for a single patient"""
        shap_values = self.explainer.shap_values(patient_data)
        return shap_values

    def waterfall_plot(self, patient_data, patient_index=0):
        """Generate waterfall plot for single prediction"""
        shap_values = self.explainer(patient_data)
        fig = shap.plots.waterfall(
            shap_values[patient_index],
            show=False
        )
        return fig

    def force_plot(self, patient_data):
        """Generate force plot"""
        shap_values = self.explainer.shap_values(patient_data)
        fig = shap.force_plot(
            self.explainer.expected_value,
            shap_values,
            patient_data,
            show=False
        )
        return fig

    def summary_plot(self):
        """Generate summary plot for all features"""
        shap_values = self.explainer.shap_values(self.X_train)
        fig = shap.summary_plot(
            shap_values,
            self.X_train,
            show=False
        )
        return fig

    def bar_plot(self):
        """Generate mean SHAP values bar plot"""
        shap_values = self.explainer(self.X_train)
        fig = shap.plots.bar(shap_values, show=False)
        return fig

    def get_top_features(self, patient_data, top_n=5):
        """Get top N contributing features for a prediction"""
        shap_values = self.explainer.shap_values(patient_data)

        if isinstance(shap_values, list):
            sv = shap_values[1]  # For binary classification
        else:
            sv = shap_values

        feature_importance = dict(
            zip(patient_data.columns, sv[0])
        )
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:top_n]

SHAP PLOTS TO GENERATE IN THE APP:

1. WATERFALL PLOT (For single patient):
   - Shows how each feature pushes the prediction
     from base value to final prediction
   - Red = pushes toward CKD, Blue = pushes away from CKD
   - This is THE MOST important plot for clinical use

2. FORCE PLOT (For single patient):
   - Horizontal visualization of feature contributions
   - Interactive in HTML format

3. SUMMARY PLOT (For overall model):
   - Shows feature importance across all patients
   - Dot plot with color indicating feature value
   - Shows which features matter most overall

4. BAR PLOT (For overall model):
   - Simple bar chart of mean absolute SHAP values
   - Quick overview of feature importance

5. DEPENDENCE PLOT (Optional):
   - Shows how a single feature affects predictions
   - Useful for features like serum creatinine

TEXT EXPLANATION FROM SHAP:
Convert SHAP values to human-readable text:
Example output:
"Your CKD risk is HIGH primarily because:

1.  Serum Creatinine (2.1 mg/dL) - 35% contribution to risk
    [Normal: 0.6-1.2 mg/dL, Yours is ELEVATED]
2.  Blood Urea (55 mg/dL) - 25% contribution to risk
    [Normal: 7-20 mg/dL, Yours is ELEVATED]
3.  Hemoglobin (9.5 gms) - 15% contribution to risk
    [Normal: 12-17 gms, Yours is LOW]
4.  Hypertension (Yes) - 12% contribution
5.  Age (62) - 8% contribution"

================================================================
SECTION 11: LLM CONVERSATIONAL INTERFACE
================================================================

FILE: src/llm/gemini_client.py
FILE: src/llm/prompts.py

PURPOSE:
Allow doctors/health workers to interact with the AI through
natural language. The LLM explains predictions, answers
questions, and provides clinical context.

SETUP:

- Use Google Gemini API (free tier: gemini-pro or
  gemini-1.5-flash)
- Get API key from: https://makersuite.google.com/app/apikey
- Store in .env file: GEMINI_API_KEY=your_key_here

IMPLEMENTATION:

# src/llm/gemini_client.py

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class RenalGuardLLM:
def **init**(self):
genai.configure(
api_key=os.getenv("GEMINI_API_KEY")
)
self.model = genai.GenerativeModel(
'gemini-1.5-flash'
)
self.chat = self.model.start_chat(history=[])
self.system_prompt = self.\_get_system_prompt()

    def _get_system_prompt(self):
        return """
        You are RenalGuard AI Assistant, a medical AI
        assistant specialized in Chronic Kidney Disease (CKD)
        screening and education.

        IMPORTANT RULES:
        1. You are NOT a doctor. Always add disclaimer.
        2. You explain AI screening results in simple language.
        3. You follow KDIGO clinical guidelines.
        4. You recommend consulting a nephrologist for
           confirmed cases.
        5. You can explain what each blood/urine test means.
        6. You provide lifestyle and dietary recommendations.
        7. You are empathetic and supportive in tone.
        8. You can explain results in layman's terms.
        9. Never diagnose. Only explain screening results.
        10. If asked about non-kidney topics, politely redirect.
        """

    def explain_prediction(self, patient_data, prediction,
                           stage, risk_score, shap_features):
        """Explain AI prediction in simple language"""
        prompt = f"""
        {self.system_prompt}

        A patient's kidney health screening results are:

        Patient Data:
        {patient_data}

        AI Screening Results:
        - CKD Risk: {prediction}
        - Estimated Stage: {stage}
        - Risk Score: {risk_score}/100

        Top Contributing Factors (from AI explanation):
        {shap_features}

        Please provide:
        1. A clear, simple explanation of what these
           results mean
        2. Which values are concerning and why
        3. Recommended next steps based on KDIGO guidelines
        4. Lifestyle modifications that may help
        5. A clear disclaimer that this is AI screening,
           not diagnosis

        Keep the language simple enough for a patient
        to understand. Use bullet points.
        """

        response = self.model.generate_content(prompt)
        return response.text

    def chat_with_doctor(self, user_message, context=None):
        """Interactive chat for follow-up questions"""
        if context:
            full_message = f"""
            Context: {context}

            User question: {user_message}

            Please answer based on the context provided.
            Always maintain medical accuracy and add
            disclaimers where appropriate.
            """
        else:
            full_message = f"""
            {self.system_prompt}

            User question: {user_message}
            """

        response = self.chat.send_message(full_message)
        return response.text

    def generate_recommendations(self, stage, risk_factors):
        """Generate personalized recommendations"""
        prompt = f"""
        {self.system_prompt}

        Based on:
        - CKD Stage: {stage}
        - Key Risk Factors: {risk_factors}

        Provide specific, actionable recommendations:
        1. Dietary changes (specific to CKD stage)
        2. Lifestyle modifications
        3. Tests to get done
        4. How often to monitor
        5. Warning signs to watch for
        6. When to seek immediate medical attention

        Follow KDIGO guidelines. Be specific and practical.
        """

        response = self.model.generate_content(prompt)
        return response.text

PROMPTS FILE:

# src/llm/prompts.py

SYSTEM_PROMPT = """...""" # As defined above

EXPLAIN_PROMPT_TEMPLATE = """..."""

CHAT_PROMPT_TEMPLATE = """..."""

RECOMMENDATION_PROMPT_TEMPLATE = """..."""

FALLBACK_RESPONSES:
If the API is unavailable (rate limit, no internet),
provide hardcoded fallback explanations:

FALLBACK_EXPLANATIONS = {
"high_risk": "Based on the screening results, there are
indicators that suggest elevated risk for kidney disease.
Key factors include elevated creatinine and blood urea
levels. We recommend consulting a nephrologist for
further evaluation. This is an AI screening result
and not a medical diagnosis.",

    "low_risk": "The screening results suggest that your
     kidney function indicators are within or near normal
     ranges. However, if you have diabetes or hypertension,
     regular monitoring is recommended. This is an AI
     screening result and not a medical diagnosis.",

    "moderate_risk": "Some indicators suggest moderate risk
     factors for kidney disease. We recommend follow-up
     testing and consultation with a healthcare provider.
     This is an AI screening result and not a medical
     diagnosis."

}

================================================================
SECTION 12: PDF CLINICAL REPORT GENERATION
================================================================

FILE: src/reports/pdf_generator.py

PURPOSE:
Generate a professional, downloadable PDF report that
summarizes the AI screening results. This report can be
printed and given to the patient or filed in medical records.

REPORT CONTENT (Page by Page):

PAGE 1 - COVER/HEADER:

- RenalGuard AI Logo
- Title: "AI-Assisted Kidney Health Screening Report"
- Subtitle: "Powered by RenalGuard AI"
- Report ID: Auto-generated (e.g., RG-2026-XXXXX)
- Date and Time of screening
- Disclaimer bar: "This is an AI-assisted screening report.
  It is NOT a medical diagnosis. Please consult a qualified
  nephrologist for clinical evaluation."

PAGE 1 - PATIENT INFORMATION SECTION:

- Patient ID (anonymized if no name given)
- Age
- Date of Screening
- Screening Location (optional)

PAGE 1 - SCREENING RESULTS:

- CKD Risk Assessment: HIGH / MODERATE / LOW
  (Color coded: Red / Orange / Green)
- Risk Score: X/100 with visual progress bar
- Estimated CKD Stage: Stage X
- Estimated eGFR: XX mL/min/1.73m²

PAGE 1 - INPUT VALUES TABLE:
| Test Parameter | Patient Value | Normal Range | Status |
|---------------|---------------|-------------|--------|
| Serum Creatinine | 2.1 mg/dL | 0.6-1.2 | ⚠️ HIGH |
| Blood Urea | 55 mg/dL | 7-20 | ⚠️ HIGH |
| Hemoglobin | 9.5 gms | 12-17 | ⚠️ LOW |
| Blood Pressure | 150 mm/Hg | 80-120 | ⚠️ HIGH |
| ... | ... | ... | ... |

PAGE 2 - AI EXPLANATION:

- SHAP waterfall plot image (embedded in PDF)
- Text explanation of top 5 contributing factors:
  "The AI model identified the following key factors
  contributing to your risk assessment:
  1.  Serum Creatinine (2.1 mg/dL) - ELEVATED
      Contribution to risk: 35%
      Normal range: 0.6-1.2 mg/dL
  2.  ..."

PAGE 2 - RECOMMENDATIONS:

- Based on KDIGO guidelines for the predicted stage
- Dietary recommendations
- Lifestyle changes
- Recommended follow-up tests
- Suggested timeline for next screening

PAGE 2 - FOOTER:

- "Generated by RenalGuard AI - AI-Assisted Screening Tool"
- "This report does not constitute medical advice or diagnosis"
- "Consult a qualified nephrologist for clinical evaluation"
- Date and Report ID

IMPLEMENTATION USING FPDF2 or REPORTLAB:

from fpdf import FPDF
import datetime

class ClinicalReportGenerator:
def **init**(self):
self.pdf = FPDF()
self.pdf.set_auto_page_break(auto=True, margin=15)

    def generate_report(self, patient_data, prediction,
                        stage, risk_score, shap_features,
                        recommendations, shap_plot_path):

        self.pdf.add_page()

        # Header
        self.pdf.set_font('Arial', 'B', 20)
        self.pdf.cell(0, 15, 'RenalGuard AI', ln=True,
                      align='C')
        self.pdf.set_font('Arial', 'I', 12)
        self.pdf.cell(0, 8,
                      'AI-Assisted Kidney Health Screening Report',
                      ln=True, align='C')

        # Disclaimer
        self.pdf.set_fill_color(255, 255, 200)
        self.pdf.set_font('Arial', 'I', 8)
        self.pdf.multi_cell(0, 5,
            'DISCLAIMER: This is an AI-assisted screening '
            'report. Not a medical diagnosis. Consult a '
            'qualified nephrologist.', fill=True)

        # ... continue building the PDF

        # Save
        report_path = f"reports/report_{datetime.datetime.now()
                        .strftime('%Y%m%d_%H%M%S')}.pdf"
        self.pdf.output(report_path)
        return report_path

================================================================
SECTION 13: STREAMLIT FRONTEND - COMPLETE UI SPECIFICATION
================================================================

FILE: app/main.py (and app/pages/\*.py)

OVERALL APP DESIGN:

- Color scheme: Medical blue (#1a73e8) + White + Light Gray
- Clean, professional, medical-grade UI
- Sidebar navigation
- Responsive layout
- Custom CSS for polished look

STREAMLIT CONFIG (.streamlit/config.toml):
[theme]
primaryColor = "#1a73e8"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

PAGE 1: HOME / LANDING PAGE
─────────────────────────────

- Hero section with project name and tagline
- Brief description of what RenalGuard AI does
- Key statistics:
  • "850M people affected by CKD worldwide"
  • "90% of cases detected too late"
  • "Our AI can help detect CKD early"
- "Get Started" button → navigates to Prediction page
- Features overview cards:
  • AI-Powered Detection
  • Explainable AI
  • Clinical Reports
  • Smart Assistant
- Tech stack badges
- How it works (step diagram)

PAGE 2: PREDICTION (Main functionality page)
─────────────────────────────────────────────
LAYOUT: Two columns

LEFT COLUMN - Input Form:

- Title: "Enter Patient Test Results"
- Input fields (all with tooltips explaining what each test is):

  Numerical inputs (st.number_input):
  • Age (10-100, default 45)
  • Blood Pressure (50-200, default 120, unit: mm/Hg)
  • Specific Gravity (1.000-1.030, default 1.020,
  step 0.005)
  • Blood Glucose Random (50-500, default 120, unit: mg/dL)
  • Blood Urea (1-200, default 40, unit: mg/dL)
  • Serum Creatinine (0.1-15.0, default 1.0,
  step 0.1, unit: mg/dL)
  • Sodium (100-170, default 140, unit: mEq/L)
  • Potassium (2.0-8.0, default 4.5, step 0.1, unit: mEq/L)
  • Hemoglobin (3.0-18.0, default 13.0, step 0.1, unit: gms)
  • Packed Cell Volume (10-60, default 40)
  • White Blood Cell Count (2000-25000, default 8000,
  unit: cells/cumm)
  • Red Blood Cell Count (2.0-8.0, default 5.0,
  step 0.1, unit: millions/cmm)

  Categorical inputs (st.selectbox):
  • Albumin (0, 1, 2, 3, 4, 5)
  • Sugar (0, 1, 2, 3, 4, 5)
  • Red Blood Cells (Normal / Abnormal)
  • Pus Cell (Normal / Abnormal)
  • Pus Cell Clumps (Present / Not Present)
  • Bacteria (Present / Not Present)
  • Hypertension (Yes / No)
  • Diabetes Mellitus (Yes / No)
  • Coronary Artery Disease (Yes / No)
  • Appetite (Good / Poor)
  • Pedal Edema (Yes / No)
  • Anemia (Yes / No)

  Submit button: "🔍 Analyze Kidney Health"

RIGHT COLUMN - Results Display (after prediction):

- Risk Level Card:
  • Large colored badge: HIGH RISK (red) / MODERATE (orange) /
  LOW (green)
  • Risk score: X/100 with circular progress indicator
  • Confidence: XX%

- CKD Stage Card:
  • "Estimated Stage: X"
  • Stage description
  • eGFR value with interpretation

- Quick Summary:
  • 3-4 bullet points of key findings

- Action buttons:
  • "📊 View Detailed Explanation" → goes to Explainability
  • "💬 Ask AI Assistant" → goes to Chat
  • "📄 Download Report" → generates and downloads PDF
  • "🔄 New Screening" → resets form

PAGE 3: EXPLAINABILITY (SHAP Visualizations)
────────────────────────────────────────────

- Title: "AI Explanation - Why This Prediction?"
- Subtitle: "Understanding what factors influenced
  the AI's assessment"

- Section 1: WATERFALL PLOT
  • Title: "Feature Contributions for This Patient"
  • SHAP waterfall plot showing how each feature pushed
  the prediction
  • Text explanation below the plot

- Section 2: TOP RISK FACTORS
  • Table or cards showing top 5 contributing factors
  • For each factor:
  - Feature name
  - Patient's value
  - Normal range
  - SHAP contribution (% and direction)
  - Status icon (✅ Normal / ⚠️ Elevated / 🔴 Critical)

- Section 3: FORCE PLOT
  • Interactive force plot (if possible in Streamlit)
  • Or static image version

- Section 4: OVERALL MODEL INSIGHTS
  • SHAP summary plot (beeswarm)
  • Feature importance bar chart
  • "These are the features that matter most OVERALL
  for kidney disease detection"

PAGE 4: AI CHAT ASSISTANT
──────────────────────────

- Title: "💬 RenalGuard AI Assistant"
- Chat interface using st.chat_message and st.chat_input
- Pre-loaded context from the latest prediction
- Suggested questions:
  • "What do my results mean?"
  • "What should I do next?"
  • "Explain my creatinine level"
  • "What dietary changes should I make?"
  • "When should I get tested again?"
- Chat history maintained in st.session_state
- Each response includes a small disclaimer

PAGE 5: DATA INSIGHTS DASHBOARD
────────────────────────────────

- Title: "📊 CKD Data Insights"
- EDA visualizations from the training data:
  • Distribution of CKD vs Non-CKD cases (pie chart)
  • Feature distributions (histograms)
  • Correlation heatmap
  • Box plots of key features by CKD status
  • CKD stage distribution (bar chart)
- Model performance section:
  • ROC Curve
  • Confusion Matrix
  • Metrics table (Accuracy, Precision, Recall, F1, AUC)
  • Model comparison chart

PAGE 6: ABOUT
─────────────

- About the project
- About the team
- About the hackathon
- Technology stack used
- Future roadmap
- References and citations
- Contact information

SIDEBAR:
────────

- RenalGuard AI logo at top
- Navigation menu:
  • 🏠 Home
  • 🔍 CKD Screening
  • 📊 AI Explanation
  • 💬 AI Assistant
  • 📈 Data Insights
  • ℹ️ About
- Divider
- "Made for AI4Dev '26 Hackathon"
- Team info
- Dark/Light mode toggle (optional)

================================================================
SECTION 14: BACKEND API (Optional - Only if time permits)
================================================================

FILE: src/api/main.py (OPTIONAL)

If time permits, create a FastAPI backend.
Otherwise, Streamlit directly loads models.

If built:

- POST /predict - Takes patient data, returns prediction
- POST /explain - Returns SHAP explanation
- POST /chat - LLM chat endpoint
- GET /report/{id} - Download PDF report
- GET /health - Health check

This is OPTIONAL. Streamlit alone is sufficient for Round 1.

================================================================
SECTION 15: GITHUB REPOSITORY REQUIREMENTS
================================================================

The GitHub repo is ONE OF THE THREE evaluation criteria.
Make it EXCELLENT.

MUST HAVE:

1. Clean, organized folder structure (as specified in Section 6)
2. Comprehensive README.md (as specified in Section 16)
3. requirements.txt with all dependencies
4. .gitignore (Python template)
5. LICENSE (MIT)
6. Proper commit history (not just one "initial commit")
7. Code comments and docstrings
8. .env.example (showing required environment variables)

.gitignore contents:
**pycache**/
_.py[cod]
_$py.class
_.so
.env
_.pkl
_.joblib
.ipynb_checkpoints/
dist/
build/
_.egg-info/
.vscode/
.idea/
venv/
env/
data/raw/_.csv
models/_.pkl
reports/_.pdf
_.log

COMMIT HISTORY (make it look professional):
Commit messages should follow conventional format:

- "feat: add CKD detection model with XGBoost"
- "feat: integrate SHAP explainability"
- "feat: add Streamlit frontend"
- "feat: implement LLM chat interface"
- "feat: add PDF report generation"
- "docs: update README with architecture diagram"
- "fix: handle missing values in preprocessing"
- "style: improve UI color scheme"
- "test: add model evaluation tests"
- "chore: update requirements.txt"

================================================================
SECTION 16: README.md SPECIFICATION
================================================================

The README is the FIRST thing judges see. Make it OUTSTANDING.

# 🩺 RenalGuard AI

> AI-Powered Early Detection & Clinical Decision Support
> for Chronic Kidney Disease

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)]
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)]
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange)]
[![License](https://img.shields.io/badge/License-MIT-yellow)]

## 🌟 Overview

RenalGuard AI is an AI-powered clinical decision support
tool designed to detect Chronic Kidney Disease (CKD) early
using basic blood and urine test parameters. With explainable
AI (SHAP), LLM-powered conversational interface, and
auto-generated clinical reports, it aims to make kidney
health screening accessible even in resource-limited settings.

**Built for AI4Dev '26 Hackathon | Healthcare & Life Sciences**

## 🎯 Problem Statement

[Include the problem statement with statistics]

## 💡 Solution

[Include solution overview with architecture diagram]

## 🏗️ Architecture

[Include architecture diagram - can be ASCII or image]

## ✨ Key Features

| Feature                 | Description                             |
| ----------------------- | --------------------------------------- |
| 🔍 CKD Detection        | Binary classification with XGBoost      |
| 📊 Stage Classification | Multi-class CKD staging                 |
| 🧠 Explainable AI       | SHAP-based prediction explanations      |
| 💬 AI Assistant         | Gemini-powered conversational interface |
| 📄 Clinical Reports     | Auto-generated PDF screening reports    |
| 📈 Data Dashboard       | Interactive EDA and model insights      |

## 🛠️ Tech Stack

[List all technologies with badges]

## 📊 Model Performance

[Include metrics table and ROC curve]

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/username/RenalGuard-AI.git
cd RenalGuard-AI
pip install -r requirements.txt
Environment Setup
Bash

cp .env.example .env
# Add your Gemini API key to .env
Run the Application
Bash

streamlit run app/main.py
📸 Screenshots
[Include screenshots of each page]

📁 Project Structure
[Include folder tree]

🔬 Model Details
Dataset
UCI Chronic Kidney Disease Dataset (400 samples)
24 clinical features
Engineered features including eGFR and CKD staging
Models
CKD Detection: XGBoost binary classifier
Stage Classification: XGBoost multi-class classifier
Explainability
SHAP (SHapley Additive exPlanations)
Waterfall plots, force plots, summary plots
Human-readable feature contribution explanations
🔮 Future Scope
ABDM (Ayushman Bharat Digital Mission) integration
Edge deployment using ONNX/TensorFlow Lite
Multi-language support for regional languages
Mobile application (React Native / Flutter)
Integration with MIMIC-IV clinical database
CKD progression prediction using time-series analysis
Federated learning for privacy-preserving model training
🏆 Hackathon
This project was built for AI4Dev '26 - National-Level
Hackathon on AI-Enabled Transformative Technologies for
Global Development, organized by PSG College of Technology.

👥 Team
[Team member names and roles]

📚 References
UCI CKD Dataset
KDIGO Clinical Practice Guidelines
SHAP: Lundberg & Lee (2017)
XGBoost: Chen & Guestrin (2016)
📄 License
MIT License

================================================================
SECTION 17: PPT CONTENT (Slide-by-Slide)
NOTE: The hackathon provides a PPT template. Use their template
but ensure these content elements are covered.

TOTAL SLIDES: 12-15 slides maximum

SLIDE 1: TITLE SLIDE

Project Name: RenalGuard AI
Tagline: "AI-Powered Early Detection & Clinical Decision
Support for Chronic Kidney Disease"
Team Name
Team Members (Name, College, Year)
Domain: Healthcare and Life Sciences
Hackathon: AI4Dev '26
SLIDE 2: THE PROBLEM

Title: "The Silent Killer: Kidney Disease"
Key statistics with visual impact:
• 850M affected worldwide
• 90% diagnosed too late
• 80% of CKD deaths in developing countries
• India: 2nd highest CKD burden
Visual: Graph showing late detection problem
Emotional hook: "By the time symptoms appear, kidneys
are already 70-85% damaged"
SLIDE 3: WHY THIS MATTERS

Cost of late detection:
• Dialysis: ₹15,000-25,000/month
• Transplant: ₹5-15 lakhs
Cost of early detection:
• Basic blood test: ₹200-500
• AI screening: Almost free
"If we detect CKD at Stage 1-2, we can PREVENT
progression to Stage 5"
SDG alignment: SDG 3 (Good Health), SDG 10
(Reduced Inequalities)
SLIDE 4: EXISTING SOLUTIONS & GAP

Table comparing existing approaches vs our solution
Existing: eGFR calculators, expensive hospital equipment,
specialist consultation
Gaps: No AI-powered screening, no explainability,
not accessible in rural areas
Our value proposition: Affordable + Explainable + Accessible
SLIDE 5: OUR SOLUTION - OVERVIEW

RenalGuard AI Architecture diagram
One-line: "Input basic test results → AI predicts CKD risk
explains why + recommends next steps"
Key capabilities listed as icons
SLIDE 6: HOW IT WORKS

Step-by-step flow:
Enter basic test results (3-4 tests minimum)
AI analyzes using trained ML models
Get risk assessment + CKD stage
View AI explanation (SHAP)
Chat with AI assistant for guidance
Download clinical report (PDF)
SLIDE 7: TECHNICAL ARCHITECTURE

Detailed architecture diagram showing:
• Data Layer (Dataset, Preprocessing)
• ML Layer (XGBoost models, SHAP)
• LLM Layer (Gemini API)
• Application Layer (Streamlit)
• Output Layer (Predictions, Reports)
Tech stack badges
SLIDE 8: AI/ML DETAILS

Models used and why
Training methodology (cross-validation)
Key metrics:
• Accuracy: XX%
• Sensitivity: XX%
• Specificity: XX%
• AUC-ROC: XX
Confusion matrix visualization
Feature importance chart
SLIDE 9: EXPLAINABLE AI (KEY DIFFERENTIATOR)

Title: "Not Just Prediction — Explanation"
SHAP waterfall plot example
"Our AI doesn't just say 'You have CKD risk'.
It explains WHY."
Example: "Your serum creatinine (2.1) contributed 35%
to the risk score"
Why this matters: Trust, clinical usability,
Responsible AI
SLIDE 10: DEMO SCREENSHOTS

Screenshots of the app:
• Input form
• Prediction results
• SHAP explanation
• Chat interface
• PDF report
Or: "See Demo Video" link
SLIDE 11: INNOVATION & UNIQUENESS

What makes us different (5 bullet points):
Multi-model approach (Detection + Staging)
Explainable AI with SHAP
LLM-powered conversational assistant
Auto-generated clinical reports
Designed for resource-limited settings
Compare with existing solutions in a table
SLIDE 12: IMPACT & SCALABILITY

Who benefits:
• Rural health clinics (PHCs, CHCs)
• ASHA workers conducting screenings
• Diabetic/hypertensive patients
• Nephrologists receiving referrals
Scalability:
• Can be integrated into existing health apps
• ABDM/Ayushman Bharat compatible (future)
• Edge deployable (ONNX)
• Minimal compute requirements
SLIDE 13: FUTURE ROADMAP

Phase 1 (Current): Working PoC ✅
Phase 2: Mobile app + regional languages
Phase 3: ABDM integration + clinical validation
Phase 4: Deployment in PHCs across India
Phase 5: Expand to other diseases
(diabetes, hypertension, liver)
SLIDE 14: TEAM & ACKNOWLEDGMENTS

Team member details with photos
Skills each member contributed
Faculty mentor (if any)
Acknowledgment to PSG Tech and AI4Dev '26
SLIDE 15: THANK YOU & Q&A

"Thank you!"
Project links: GitHub, Demo, Video
Contact information
QR code to GitHub repo
================================================================
SECTION 18: DEMO VIDEO SCRIPT
TOTAL DURATION: 3-5 minutes maximum
FORMAT: Screen recording with voiceover
TOOL: OBS Studio (free) or Loom (free)

SCRIPT:

[0:00 - 0:20] INTRO
"Hello! We are [team name] and this is RenalGuard AI —
an AI-powered early detection and clinical decision support
platform for Chronic Kidney Disease. Built for the AI4Dev 2026
hackathon under the Healthcare and Life Sciences domain."

[0:20 - 0:50] PROBLEM
"850 million people worldwide suffer from Chronic Kidney
Disease, yet 90% are diagnosed too late. By the time symptoms
appear, kidneys are already 70-85% damaged. Our solution
addresses this by enabling early detection using just basic
blood and urine test results."

[0:50 - 1:30] SHOW THE APP - HOME PAGE

Show the landing page
Briefly highlight key features
Click on "CKD Screening"
[1:30 - 2:30] LIVE DEMO - PREDICTION
"Let me demonstrate with a sample patient. I'll enter some
test values..."

Enter sample values (use a HIGH RISK example)
Click "Analyze"
Show the prediction results
"As you can see, the AI has detected HIGH RISK with a
risk score of 78/100 and estimated Stage 3."
[2:30 - 3:30] SHAP EXPLANATION
"What makes RenalGuard AI unique is its explainability."

Navigate to Explanation page
Show SHAP waterfall plot
"The AI explains that serum creatinine at 2.1 mg/dL
contributed 35% to the risk, followed by blood urea
at 55 mg/dL contributing 25%..."
"This transparency builds trust with healthcare
professionals."
[3:30 - 4:00] LLM CHAT
"Doctors can also interact with our AI assistant..."

Show a quick chat interaction
Ask "What do these results mean?"
Show the response
[4:00 - 4:20] PDF REPORT
"The platform also generates a downloadable clinical report..."

Show the PDF generation
Quick scroll through the report
[4:20 - 5:00] CLOSING
"RenalGuard AI demonstrates how AI can be used to make
healthcare more accessible, especially in resource-limited
settings. With explainable AI, conversational interfaces,
and clinical reporting, we aim to bridge the gap between
AI and clinical practice.

Future plans include ABDM integration, edge deployment,
and multi-language support.

Thank you for watching. We are [team name] and this is
RenalGuard AI."

================================================================
SECTION 19: DEPLOYMENT INSTRUCTIONS
OPTION 1: STREAMLIT CLOUD (Recommended - Easiest)

Push code to GitHub
Go to share.streamlit.io
Connect GitHub account
Select repository: RenalGuard-AI
Set main file: app/main.py
Add secrets:
GEMINI_API_KEY = "your_key"
Deploy
Get URL like: https://renalguard-ai.streamlit.app
OPTION 2: HUGGING FACE SPACES

Create account on huggingface.co
Create new Space (Streamlit SDK)
Upload all files
Add requirements.txt
Add secrets in Settings
Auto-deploys
OPTION 3: LOCAL ONLY (If deployment issues)

Just ensure the judges can run:
git clone <repo>
cd RenalGuard-AI
pip install -r requirements.txt
streamlit run app/main.py

Include these instructions clearly in README.

================================================================
SECTION 20: MEDICAL DOMAIN KNOWLEDGE
INCLUDE THIS KNOWLEDGE IN THE APP AND PPT:

CKD RISK FACTORS:

Diabetes (Type 1 and Type 2) - #1 cause
Hypertension - #2 cause
Family history of kidney disease
Age > 60
Obesity
Smoking
History of acute kidney injury
Cardiovascular disease
Prolonged use of NSAIDs
KEY BIOMARKERS FOR CKD:

Serum Creatinine:

Normal: 0.6-1.2 mg/dL (male), 0.5-1.1 mg/dL (female)
Elevated = reduced kidney function
Most important single marker
Blood Urea Nitrogen (BUN):

Normal: 7-20 mg/dL
Elevated = kidneys not filtering urea properly
eGFR (estimated Glomerular Filtration Rate):

Normal: >90 mL/min/1.73m²
Calculated from creatinine, age, sex, race
Gold standard for kidney function assessment
Urine Albumin:

Normal: 0
Presence indicates kidney damage (albuminuria)
Persistent albuminuria = CKD marker
Hemoglobin:

Low hemoglobin in CKD = kidneys not producing
enough erythropoietin
Anemia is common in CKD
Electrolytes (Sodium, Potassium):

Abnormal levels indicate electrolyte imbalance
due to kidney dysfunction
KDIGO RECOMMENDATIONS BY STAGE:

Stage 1-2: Lifestyle changes, control BP and blood sugar,
monitor annually
Stage 3: Monitor every 3-6 months, may need medication,
dietary changes
Stage 4: Frequent monitoring, prepare for possible dialysis,
referral to nephrologist
Stage 5: Dialysis or transplant required, immediate
specialist care
DIETARY RECOMMENDATIONS FOR CKD:

Reduce sodium (< 2g/day)
Control protein intake (0.6-0.8g/kg/day in advanced stages)
Limit potassium (in stages 4-5)
Limit phosphorus
Stay hydrated (but not excessive in later stages)
Avoid processed foods
================================================================
SECTION 21: EVALUATION CRITERIA ALIGNMENT
Map every feature to evaluation criteria:

ROUND 1 CRITERIA:

"Project PPT quality and clarity"
→ Professional PPT with clear problem, solution,
architecture, demo, and impact
→ Use the hackathon's template
→ 12-15 slides, not too text-heavy
→ Include diagrams and visuals

"GitHub code quality and documentation"
→ Clean folder structure (Section 6)
→ Comprehensive README.md (Section 16)
→ Code comments and docstrings
→ requirements.txt
→ .gitignore
→ Professional commit history
→ Jupyter notebooks for EDA and training

"Demo video clarity and presentation"
→ 3-5 minute video (Section 18)
→ Clear voiceover
→ Show all key features
→ Good audio and video quality
→ Professional intro and outro

ROUND 2 CRITERIA (prepare for these):

"Presentation and creativity"
→ Creative UI/UX
→ Engaging presentation style
→ Live demo capability

"Solution viability, real world issues"
→ Show how this works in real clinics
→ Address practical deployment challenges
→ ABDM integration concept
→ Edge deployment concept

"Originality of the solution"
→ SHAP explainability (not common in CKD projects)
→ LLM conversational interface (novel)
→ PDF clinical reports (practical)
→ Multi-model approach (detection + staging)

================================================================
SECTION 22: FILE-BY-FILE CODE SPECIFICATIONS
Below are detailed specifications for EVERY file to be created.

FILE 1: app/main.py
───────────────────

Entry point for Streamlit application
Import all pages
Set page config (title, icon, layout)
Create sidebar navigation
Route to appropriate page based on selection
Initialize session state variables
Load models at startup (cache with @st.cache_resource)
Key session state variables:

st.session_state.prediction (dict with results)
st.session_state.patient_data (dict with input values)
st.session_state.shap_values (SHAP output)
st.session_state.chat_history (list of messages)
st.session_state.page (current page)
FILE 2: app/pages/prediction.py
────────────────────────────────

Two-column layout
Left: Input form with all 24 features
Right: Results display
On submit:
Validate inputs
Preprocess (scale, encode)
Run CKD detection model
If CKD positive: Run stage classification
Calculate eGFR
Calculate risk score (0-100)
Generate SHAP values
Store everything in session state
Display results with color coding
Risk score calculation:
risk_score = int(prediction_probability * 100)
Where prediction_probability is the probability from
model.predict_proba()

FILE 3: app/pages/explainability.py
────────────────────────────────────

Display SHAP plots from session state
Waterfall plot (matplotlib figure embedded in Streamlit)
Top features table with explanations
Summary plot
Feature importance bar chart
Text explanations generated from SHAP values
FILE 4: app/pages/chat.py
──────────────────────────

Chat UI using st.chat_message and st.chat_input
Load prediction context from session state
Initialize Gemini client
Handle user messages
Display AI responses
Suggested question buttons
Error handling for API failures (show fallback)
FILE 5: app/components/input_form.py
─────────────────────────────────────

Reusable function that creates the input form
Returns a dictionary of patient data
Input validation (ranges, types)
Tooltips for each field explaining normal ranges
"Use Sample Data" button for demo purposes
(pre-fills with a high-risk patient profile)
SAMPLE DATA (for demo):
sample_high_risk = {
'age': 62, 'bp': 150, 'sg': 1.010, 'al': 3, 'su': 2,
'rbc': 'abnormal', 'pc': 'abnormal',
'pcc': 'present', 'ba': 'notpresent',
'bgr': 250, 'bu': 55, 'sc': 2.1, 'sod': 130,
'pot': 5.5, 'hemo': 9.5, 'pcv': 30,
'wc': 11000, 'rc': 3.8,
'htn': 'yes', 'dm': 'yes', 'cad': 'no',
'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
}

sample_low_risk = {
'age': 35, 'bp': 120, 'sg': 1.020, 'al': 0, 'su': 0,
'rbc': 'normal', 'pc': 'normal',
'pcc': 'notpresent', 'ba': 'notpresent',
'bgr': 110, 'bu': 15, 'sc': 0.9, 'sod': 140,
'pot': 4.2, 'hemo': 14.5, 'pcv': 44,
'wc': 7500, 'rc': 5.2,
'htn': 'no', 'dm': 'no', 'cad': 'no',
'appet': 'good', 'pe': 'no', 'ane': 'no'
}

FILE 6: src/data/preprocessor.py
─────────────────────────────────
Complete preprocessing pipeline as described in Section 7.
Should include:

class DataPreprocessor with methods:
load_data(filepath)
clean_data(df)
handle_missing_values(df)
encode_categoricals(df)
engineer_features(df)
scale_features(df)
preprocess_single_patient(patient_dict)
← For real-time prediction
get_train_test_split(df, test_size=0.2)
FILE 7: src/models/ckd_detector.py
───────────────────────────────────

class CKDDetector:
init(self)
train(X_train, y_train)
predict(X)
predict_proba(X)
evaluate(X_test, y_test) → returns dict of metrics
save(filepath)
load(filepath) → class method
get_feature_importance()
FILE 8: src/models/stage_classifier.py
───────────────────────────────────────

class StageClassifier:
init(self)
train(X_train, y_train)
predict(X)
predict_proba(X)
evaluate(X_test, y_test)
save(filepath)
load(filepath)
FILE 9: src/models/model_trainer.py
────────────────────────────────────

Main training script that:
Loads and preprocesses data
Trains all models
Evaluates all models
Compares models
Saves best models
Saves metrics to JSON
Generates evaluation plots
Can be run standalone: python -m src.models.model_trainer
FILE 10: src/explainability/shap_explainer.py
──────────────────────────────────────────────
Complete SHAP integration as described in Section 10.

FILE 11: src/llm/gemini_client.py
──────────────────────────────────
Complete LLM integration as described in Section 11.

FILE 12: src/reports/pdf_generator.py
──────────────────────────────────────
Complete PDF generation as described in Section 12.

================================================================
SECTION 23: TESTING REQUIREMENTS
FILE: tests/test_models.py

Write basic tests to show code quality:

import pytest
from src.models.ckd_detector import CKDDetector
from src.data.preprocessor import DataPreprocessor

class TestDataPreprocessor:
def test_load_data(self):
"""Test that data loads correctly"""
preprocessor = DataPreprocessor()
df = preprocessor.load_data("data/raw/kidney_disease.csv")
assert df is not None
assert len(df) == 400
assert len(df.columns) == 25

text

def test_clean_target(self):
    """Test target variable cleaning"""
    # Test that ckd\t becomes ckd, etc.
    pass

def test_missing_values_handled(self):
    """Test no missing values after preprocessing"""
    pass
class TestCKDDetector:
def test_model_trains(self):
"""Test that model trains without errors"""
pass

text

def test_prediction_output(self):
    """Test prediction returns valid output"""
    pass

def test_probability_range(self):
    """Test probabilities are between 0 and 1"""
    pass
class TestSHAPExplainer:
def test_shap_values_generated(self):
"""Test SHAP values are generated"""
pass

text

def test_top_features(self):
    """Test top features extraction"""
    pass
================================================================
SECTION 24: FINAL CHECKLIST
Before submission on March 8, verify ALL of these:

GITHUB REPOSITORY:
[ ] Clean folder structure matching Section 6
[ ] README.md is comprehensive (Section 16)
[ ] requirements.txt has all dependencies
[ ] .gitignore is set up
[ ] LICENSE file exists (MIT)
[ ] .env.example exists (without actual keys)
[ ] Code has comments and docstrings
[ ] Notebooks are clean and runnable
[ ] No sensitive data committed (API keys, passwords)
[ ] Multiple meaningful commits (not just one)
[ ] Repository is PUBLIC

CODE:
[ ] Data preprocessing pipeline works
[ ] CKD detection model trains and predicts
[ ] Stage classification model works
[ ] SHAP explainability generates plots
[ ] LLM chat works (with fallback for no API key)
[ ] PDF report generates correctly
[ ] Streamlit app runs without errors
[ ] All pages are functional
[ ] Sample data buttons work for demo
[ ] Error handling exists for edge cases

STREAMLIT APP:
[ ] App loads without errors
[ ] All pages accessible via sidebar
[ ] Input form works with validation
[ ] Prediction results display correctly
[ ] SHAP plots render in Streamlit
[ ] Chat interface works
[ ] PDF download works
[ ] Mobile-responsive (test narrow browser)
[ ] Professional look and feel

MODELS:
[ ] Model pickle files are saved
[ ] Scaler pickle file is saved
[ ] Model metadata JSON exists
[ ] Evaluation metrics are documented
[ ] Model comparison is done

PPT:
[ ] Uses hackathon template (if provided)
[ ] 12-15 slides
[ ] Problem clearly defined
[ ] Solution clearly explained
[ ] Architecture diagram included
[ ] Demo screenshots included
[ ] Model metrics included
[ ] Impact and scalability addressed
[ ] Future roadmap included
[ ] Team info included
[ ] No spelling/grammar errors

DEMO VIDEO:
[ ] 3-5 minutes duration
[ ] Clear audio (no background noise)
[ ] Screen recording is clear and readable
[ ] All key features demonstrated
[ ] Voiceover explains each feature
[ ] Professional intro and outro
[ ] Uploaded to YouTube/Drive/Loom
[ ] Link is accessible (not private)

SUBMISSION:
[ ] PPT uploaded on hackathon website
[ ] GitHub link submitted
[ ] Demo video URL submitted
[ ] All team members registered
[ ] Domain selected: Healthcare and Life Sciences
[ ] Submission before March 8, 2026 deadline

================================================================
END OF PROJECT BRIEF
```
