<div align="center">
  <img src="assets/home.png" alt="RenalGuard AI Logo" width="100%">
  
  <h1>🩺 RenalGuard AI</h1>
  <p><strong>An Intelligent Clinical Decision Support System for Early Chronic Kidney Disease Detection</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 🏆 Project Overview

Chronic Kidney Disease (CKD) affects over **850 million people worldwide**, yet nearly **90% of individuals are diagnosed only in the late stages** when irreversible damage has occurred. 

**RenalGuard AI** bridges this critical screening gap. It is an AI-powered diagnostic workstation designed for resource-limited healthcare settings. By analyzing 24 basic blood and urine parameters, our system detects CKD risk instantly, explains the exact clinical reasons behind the prediction, and generates professional referral reports.

### The Problem
* **Late Diagnosis:** Symptoms often appear only when kidney function has dropped below 20%.
* **High Costs:** Late-stage treatment (dialysis/transplant) is financially devastating.
* **Specialist Shortage:** Primary care centers lack the predictive tools to catch CKD early.

### The Solution
* **Ultra-Early Detection:** Machine learning identifies subtle biomarker patterns long before physical symptoms emerge.
* **Explainable AI:** We reject "black-box" healthcare. Our models explain *exactly* which biomarkers (e.g., Creatinine, Hemoglobin) drove the risk assessment.
* **Accessible Interface:** A unified, medical-grade dashboard operable by any healthcare worker.

---

## ✨ Features

- **🧠 Intelligent Risk Stratification**: Binary classification of CKD presence and multi-class estimation of the specific CKD Stage (1-5).
- **📊 Real-Time eGFR Calculation**: Automatically computes the estimated Glomerular Filtration Rate based on patient age and serum creatinine.
- **🔍 Clinical Interpretability**: Uses SHAP (SHapley Additive exPlanations) translated into plain English to tell doctors and patients exactly *why* a risk was flagged.
- **💬 Virtual AI Consult**: An integrated LLM Assistant that answers immediate patient questions about dietary changes and test results.
- **📄 1-Click Medical Reporting**: Generates verifiable, professional PDF reports complete with clinical disclaimers for patient medical records.

---

## 🏗️ System Architecture & ML Pipeline

RenalGuard AI is built on a robust, clinically-aligned machine learning pipeline using data validated by the UCI Machine Learning Repository.

1. **Data Ingestion & Cleaning**: Intelligent imputation using K-Nearest Neighbors (KNN) to handle missing lab results typical in clinical settings.
2. **Feature Engineering**: Derivation of critical clinical ratios and eGFR integration.
3. **Ensemble Modeling**: Stacking top-performing gradient boosting models (`XGBoost`, `LightGBM`) to maximize ROC-AUC and minimize false negatives.
4. **Explainability Layer**: SHAP tree explainers map complex ensemble decisions back to human-readable physiological factors.
5. **Presentation Layer**: A seamless, single-page Streamlit application designed as a professional medical workstation.

---

## 📸 Application Showcase

### 1. Clinical Workstation Dashboard
The unified screening intake and risk assessment interface.
![Clinical Intake Dashboard](assets/insights.png)

### 2. High-Precision Diagnostic Results
Instant risk grading and calculated glomerular filtration rate (eGFR).
![Diagnostic Analysis](assets/result.png)

### 3. Transparent AI & Interpretability
Plain-English explanations of key risk factors driving the diagnosis.
![Model Interpretability](assets/shap.png)

---

## 🚀 Getting Started

To run RenalGuard AI locally, follow these steps:

### Prerequisites
- Python 3.9 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jeseem24/RenalGuard-AI.git
   cd RenalGuard-AI
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Clinical Workstation:**
   ```bash
   streamlit run app/main.py
   ```

---

## 🔬 Dataset & Methodology

This project utilizes the **Chronic Kidney Disease dataset** from the UCI Machine Learning Repository (originating from Apollo Hospitals, Tamil Nadu). It contains 400 patient records with 24 distinct clinical features including:

- **Blood Chemistry**: Blood Urea, Serum Creatinine, Sodium, Potassium, Hemoglobin
- **Urine Analysis**: Specific Gravity, Albumin, Sugar, Red/White Blood Cells
- **Vitals & Demographics**: Age, Blood Pressure
- **Comorbidities**: Hypertension, Diabetes Mellitus, Coronary Artery Disease

---

## ⚕️ Medical Disclaimer

*RenalGuard AI is a proof-of-concept clinical decision support tool designed for hackathon demonstration purposes. It is **not** a certified medical device and should **never** replace professional medical judgment, diagnosis, or treatment. Always consult a qualified healthcare provider regarding medical conditions.*

---
<div align="center">
  <p>Built for the <strong>AI4Dev '26 Hackathon</strong> | Healthcare and Life Sciences Track</p>
</div>
