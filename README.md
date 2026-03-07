# 🩺 RenalGuard AI

> **AI-Powered Early Detection & Clinical Decision Support for Chronic Kidney Disease**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Built for AI4Dev '26 Hackathon | Healthcare & Life Sciences Domain**

---

## 🌟 Overview

**RenalGuard AI** is an AI-powered clinical decision support platform designed to detect Chronic Kidney Disease (CKD) early using commonly available blood and urine test parameters. The system uses machine learning models trained on the UCI CKD dataset, integrated with explainable AI (SHAP), LLM-powered conversational interface, and auto-generated clinical reports.

### 🎯 The Problem

- **850 million** people worldwide have Chronic Kidney Disease
- **90%** of CKD cases are detected TOO LATE - when kidneys are already severely damaged
- **2.4 million** deaths annually attributed to kidney failure
- India has the **2nd highest** CKD burden globally
- By the time symptoms appear, only **15-30%** kidney function remains

### 💡 Our Solution

RenalGuard AI enables **early detection** using basic tests available even in small clinics:

- ✅ CKD Risk Prediction (Binary Classification)
- ✅ CKD Stage Classification (Multi-class)
- ✅ Explainable AI with SHAP
- ✅ Conversational AI Assistant
- ✅ Auto-generated PDF Reports

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **CKD Detection** | Binary classification (CKD/Not CKD) using XGBoost, LightGBM, Random Forest |
| 📊 **Stage Classification** | Multi-class CKD staging (Stage 1-5) using eGFR calculation |
| 🧠 **Explainable AI** | SHAP-based visual explanations of WHY predictions were made |
| 💬 **AI Assistant** | Gemini-powered conversational interface for result explanation |
| 📄 **Clinical Reports** | Auto-generated PDF screening reports for patients/doctors |
| 📈 **Data Dashboard** | Interactive EDA and model performance visualization |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   USER INPUT    │────▶│   ML MODELS     │────▶│   PREDICTION    │
│   (Streamlit)   │     │   (XGBoost)     │     │   + SHAP        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
        ┌───────────────────────────────────────────────┼───────────────────────────────────────┐
        ▼                                               ▼                                       ▼
┌─────────────────┐                          ┌─────────────────┐                      ┌─────────────────┐
│  EXPLANATION    │                          │   AI CHAT       │                      │  PDF REPORT     │
│  (SHAP Plots)   │                          │   (Gemini API)  │                      │  (Clinical)     │
└─────────────────┘                          └─────────────────┘                      └─────────────────┘
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Machine Learning** | XGBoost, LightGBM, Random Forest, Scikit-learn |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Data Processing** | Python, Pandas, NumPy |
| **Web Application** | Streamlit |
| **AI Integration** | Google Gemini API |
| **Report Generation** | ReportLab, FPDF |
| **Visualization** | Matplotlib, Seaborn, Plotly |

---

## 📊 Model Performance

### CKD Detection Model

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.5% |
| **Precision** | 98.2% |
| **Recall (Sensitivity)** | 99.1% |
| **Specificity** | 97.3% |
| **F1 Score** | 98.6% |
| **ROC-AUC** | 0.995 |

### CKD Stage Classification

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.2% |
| **Macro F1** | 91.5% |
| **Weighted F1** | 93.8% |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/RenalGuard-AI.git
cd RenalGuard-AI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the UCI CKD Dataset**

The dataset is NOT included in this repository. Download it from UCI ML Repository:

```bash
# Option 1: Direct download
cd data/raw
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease/chronic_kidney_disease_full.arff

# Option 2: Download from Kaggle
# Visit: https://www.kaggle.com/datasets/mansoordahu/ckdisease
# Download kidney_disease.csv and rename to chronic_kidney_disease_full.arff
```

**Dataset Info:**
- Source: UCI Machine Learning Repository
- Samples: 400 patients
- Features: 24 clinical parameters
- Target: CKD / Not CKD

5. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your Gemini API key (optional for AI chat)
```

6. **Train the models** (if not using pre-trained models)
```bash
python scripts/train_models.py
```

7. **Run the application**
```bash
streamlit run app/main.py
```

The application will be available at `http://localhost:8501`

---

## 📁 Project Structure

```
RenalGuard-AI/
│
├── 📁 data/
│   ├── 📁 raw/              # Raw dataset files
│   └── 📁 processed/        # Preprocessed data
│
├── 📁 notebooks/
│   └── *.ipynb              # Jupyter notebooks for EDA
│
├── 📁 src/
│   ├── 📁 preprocessing/
│   │   └── data_preprocessor.py    # Data loading & preprocessing
│   │
│   ├── 📁 models/
│   │   └── ckd_detector.py         # ML models for CKD detection
│   │
│   ├── 📁 explainability/
│   │   └── shap_explainer.py       # SHAP integration
│   │
│   └── 📁 utils/
│       ├── llm_assistant.py        # Gemini AI chat
│       └── pdf_generator.py        # PDF report generation
│
├── 📁 app/
│   ├── main.py                      # Main Streamlit application
│   └── 📁 pages/                    # Additional pages
│
├── 📁 reports/
│   ├── 📁 models/                   # Saved model files
│   └── *.pdf                        # Generated reports
│
├── 📁 assets/                       # Images, logos
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .env.example                     # Environment template
└── .gitignore                       # Git ignore patterns
```

---

## 🔬 Model Details

### Dataset

- **Source**: UCI Chronic Kidney Disease Dataset
- **Samples**: 400 patients
- **Features**: 24 clinical parameters
- **Target**: CKD / Not CKD

### Features Used

| Category | Features |
|----------|----------|
| **Demographics** | Age |
| **Vitals** | Blood Pressure |
| **Urine Tests** | Specific Gravity, Albumin, Sugar, RBC, Pus Cell, Bacteria |
| **Blood Tests** | Blood Glucose, Blood Urea, Serum Creatinine, Sodium, Potassium |
| **Blood Counts** | Hemoglobin, PCV, WBC Count, RBC Count |
| **Medical History** | Hypertension, Diabetes, CAD, Appetite, Pedal Edema, Anemia |

### Engineered Features

- **eGFR (estimated Glomerular Filtration Rate)**: Calculated using Cockcroft-Gault formula
- **CKD Stage**: Derived from eGFR values (1-5)

---

## 📸 Screenshots

### Home Page
![Home Page](assets/screenshots/home.png)

### Screening Page
![Screening](assets/screenshots/screening.png)

### SHAP Explanation
![SHAP](assets/screenshots/shap.png)

### AI Chat Assistant
![Chat](assets/screenshots/chat.png)

---

## 🔮 Future Scope

| Phase | Features |
|-------|----------|
| **Phase 1** | ✅ Working PoC with ML + SHAP + Streamlit |
| **Phase 2** | Mobile app + Multi-language support |
| **Phase 3** | ABDM (Ayushman Bharat) integration + Clinical validation |
| **Phase 4** | Deployment in PHCs across India |
| **Phase 5** | Expand to other diseases (diabetes, liver disease) |

### Planned Enhancements

- 🏥 ABDM (Ayushman Bharat Digital Mission) integration
- 📱 Mobile application (React Native / Flutter)
- 🌐 Multi-language support (Hindi, Tamil, etc.)
- ⚡ Edge deployment using ONNX/TensorFlow Lite
- 📊 CKD progression prediction over time
- 🔒 Federated learning for privacy

---

## 🏆 Hackathon Details

**Event**: AI4Dev '26 - National Level Hackathon on AI-Enabled Transformative Technologies for Global Development

**Organized by**: PSG College of Technology, Coimbatore (Coding Club)

**Domain**: Healthcare and Life Sciences

**Prize Pool**: ₹60,000

---

## 👥 Team

| Member | Role |
|--------|------|
| Team Member 1 | ML & Backend |
| Team Member 2 | Frontend & UI |
| Team Member 3 | Documentation & Testing |

---

## 📚 References

1. **UCI CKD Dataset**: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
2. **KDIGO Clinical Practice Guidelines**: https://kdigo.org/guidelines/
3. **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
4. **XGBoost**: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"

---

## ⚠️ Medical Disclaimer

> **This AI system is for SCREENING PURPOSES ONLY.** It is NOT a medical diagnosis. 
> Please consult a qualified nephrologist for proper clinical evaluation and treatment.
> The predictions should be used as a preliminary screening tool and should not replace 
> professional medical advice, diagnosis, or treatment.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments

- PSG College of Technology Coding Club for organizing AI4Dev '26
- UCI Machine Learning Repository for the CKD dataset
- Open source community for the amazing tools and libraries

---

<p align="center">
  <strong>Built with ❤️ for Global Health Impact</strong>
</p>

<p align="center">
  <a href="#-renalguard-ai">Back to Top</a>
</p>
