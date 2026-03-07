# 🩺 RenalGuard AI
> **"Turning Clinical Data into Categorical Certainty"**

**AI-Powered Early Detection & Clinical Decision Support for Chronic Kidney Disease (CKD)**

![RenalGuard Hero](assets/home.png)

---

## 🏆 Project Objectives
- **Target**: AI4Dev '26 Hackathon (Healthcare & Life Sciences)
- **Problem**: Chronic Kidney Disease (CKD) affects 850 million people. 90% of cases are detected in late stages when intervention is less effective.
- **Solution**: A comprehensive clinical support system that leverages machine learning to detect CKD using 24 common clinical biomarkers.
- **Impact**: Provides a low-cost, high-precision screening tool for primary healthcare centers and resource-constrained clinical settings.

---

## 🚀 The Solution: Systematic Diagnostic Workflow

RenalGuard AI streamlines the clinical analysis process through a structured 4-phase architecture:

### 1. Patient Screening Intelligence
A digitized clinical intake interface that captures comprehensive patient history, including vitals, blood chemistry, and urine analysis.
![Clinical Intake Dashboard](assets/insights.png)

### 2. High-Precision Machine Learning Engine
The system utilizes a Gradient Boosted Ensemble (XGBoost & LightGBM) to achieve **98.5% Accuracy** in binary classification and provides automated CKD Staging (Stage 1-5) based on eGFR calculations.
![Diagnostic Analysis](assets/result.png)

### 3. Explainable AI & Transparency (SHAP)
To build clinical trust, the system incorporates **SHAP Interpretability**. This visualizes the mathematical contribution of specific biomarkers (e.g., Creatinine, Hemoglobin) to the specific patient's risk profile.
![Model Interpretability](assets/shap.png)

### 4. Automated Clinical Documentation
Generates professional-grade screening reports in PDF format, summarizing diagnostic results, feature importance, and evidence-based clinical recommendations.

---

## 🛠️ Technical Implementation & Methodology

The platform is built on a robust data science pipeline, prioritizing clinical accuracy and interpretability.

- **Interface**: Streamlit with a customized Professional Medical UI (Optimized for clarity and focus).
- **Core Analytics**: Gradient Boosting Ensembles (Optimized for structured clinical data).
- **Interpretability**: SHAP (Shapley Additive exPlanations) for model transparency.
- **Conversational Support**: LLM-integrated clinical assistant (Gemini Pro) for query handling.
- **Data Source**: UCI Chronic Kidney Disease Dataset (Verified clinical records).

---

## 📂 Project Architecture

```bash
RenalGuard-AI/
├── 📁 app/             # Main Presentation Layer & UI Logic
├── 📁 src/             # Core Analytical Engine (Models, Preprocessing, XAI)
├── 📁 data/            # Clinical Datasets
├── 📁 reports/         # Clinical Document Generation Output
├── 📁 docs/assets/     # System Visualizations & Technical Documentation
└── requirements.txt    # Production Environment Dependencies
```

---

## 🏁 Operational Deployment

1. **System Initialization**
   ```bash
   git clone https://github.com/Jeseem24/RenalGuard-AI.git
   cd RenalGuard-AI
   pip install -r requirements.txt
   ```

2. **Launch Analytical Interface**
   ```bash
   streamlit run app/main.py
   ```

3. **Evaluation Protocol**: Navigate to the **CKD Screening** module, enter patient clinical parameters, and execute the diagnostic analysis to generate a risk profile.

---

## ⚖️ Medical Disclaimer
*RenalGuard AI is an AI-assisted screening tool, not a diagnostic medical device. It is intended to assist clinicians in early detection and should be used alongside professional medical evaluation and laboratory verification.*

---

<p align="center">
  <strong>Built with ❤️ for Global Health Tech Hackathon '26</strong>
</p>
