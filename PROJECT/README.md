# 🛡️ AML Sentinel: Anti-Money Laundering Detection System
### End-to-End Machine Learning System for Financial Transaction Risk Monitoring & Compliance

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Course](https://img.shields.io/badge/KL%20University-Machine%20Learning%20(B.Tech)-blue)](https://www.kluniversity.in/)

---

## 👥 Student Credentials & Project Details
* **Project Title**: Anti-Money Laundering Detection Using Machine Learning for Financial Transaction Analysis
* **Students**: 
  - **K. VISHNU** (Roll No: `2520030199`)
  - **U. VIDHATH** (Roll No: `2520030597`)
* **Guide**: **NIRMALAJYOTHI NARISETTY** (Department of Computer Science & Engineering)
* **University**: **KL UNIVERSITY** (A.Y.: 2026-27, 4th Semester B.Tech)

---

## 🚀 Key Features & Capabilities

1. **Production Web Application & Screener (`app.py`)**:
   - Real-time single transaction screening with instant circular Risk Gauge (0-100%).
   - Risk Tier Classification: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL RISK`.
   - Automated AML Typology Explanations (Structuring detection, Z-score baseline deviation, velocity spikes).
   - Test presets: Normal retail, Structuring/Smurfing, Rapid Layering, Corporate transfer.
2. **Batch Transaction Screening & Compliance Audit Export**:
   - Process thousands of records via CSV upload.
   - Interactive filtering (Approved vs Flagged Suspicious).
   - One-click CSV export: `AML_Compliance_Audit_Report.csv`.
3. **Customer Account Profiler**:
   - Search accounts (1 - 1000) to inspect baseline spending averages, volatility ($\sigma$), and historical transaction counts.
4. **Zero Data Leakage Pipeline**:
   - `StandardScaler` fitted strictly on the 80% training split prior to transforming the 20% test split.
5. **Ensemble Machine Learning Architecture**:
   - Trained and benchmarked 6 algorithms: **Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and KNN**.
   - Champion Model: **Gradient Boosting Classifier (99.86% Accuracy, 0.9386 F1-Score, 0.9995 ROC-AUC)**.
   - Tuned XGBoost: **97.85% Recall** (catches nearly all laundering activities).

---

## 📊 Benchmark Model Performance Comparison

| Algorithm Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Champion)** | **99.86%** | **95.96%** | **91.85%** | **0.9386** | **0.9995** | 🏆 **Production Model** |
| **Tuned XGBoost (GridSearchCV)** | **99.82%** | **88.03%** | **97.85%** | **0.9268** | **0.9999** | Highest Recall |
| **Random Forest Classifier** | 99.80% | 90.04% | 93.13% | 0.9156 | 0.9997 | Ensemble Bagging |
| **Tuned Random Forest** | 99.78% | 89.54% | 91.85% | 0.9068 | 0.9997 | Optimized Depth |
| **Decision Tree Classifier** | 99.52% | 72.17% | 95.71% | 0.8229 | 0.9779 | Single Tree Baseline |
| **K-Nearest Neighbors (k=5)** | 98.98% | 68.83% | 22.75% | 0.3419 | 0.8869 | Distance-based |
| **Logistic Regression (Balanced)** | 95.18% | 19.43% | 99.57% | 0.3252 | 0.9951 | Linear Baseline |

---

## 📂 Project Structure

```text
Anti-Money laundering project/
│
├── app.py                                # Production Flask Web Application
├── run_app.bat                           # 🚀 ONE-CLICK Windows Double-Click Launcher
├── requirements.txt                      # Pinned Python Dependencies
├── README.md                             # Comprehensive Documentation
│
├── data/
│   └── financial_transactions.csv        # 100,000 Financial Transactions Dataset
│
├── models/                               # Serialized Production Artifacts
│   ├── best_aml_model.joblib             # Trained Gradient Boosting Model
│   ├── scaler.joblib                     # Fitted Zero-Leakage StandardScaler
│   ├── customer_profiles.json            # 1,000 Customer Baseline Profiles
│   ├── feature_metadata.json             # Feature Ordering & Threshold Metadata
│   └── metrics.json                      # Serialized Evaluation Metrics
│
├── notebook/                             # 📚 Complete Suite of Executed Notebooks
│   ├── AML_Detection_Complete_Project.ipynb  # ⭐ Master Submission (All 10 Instructor Sections)
│   ├── 01_Dataset_Exploration_and_Audit.ipynb
│   ├── 02_Exploratory_Data_Analysis_EDA.ipynb
│   ├── 03_Data_Preprocessing_and_Cleaning.ipynb
│   ├── 04_Feature_Engineering_and_Typologies.ipynb
│   ├── P3_SLR.ipynb                      # Practical 3: Simple Linear Regression
│   ├── P3_MLR.ipynb                      # Practical 3: Multiple Linear Regression
│   ├── P3_GD.ipynb                       # Practical 3: Gradient Descent from Scratch
│   ├── 08_Classification_Models_and_Baselines.ipynb
│   ├── 09_Ensemble_Learning_RandomForest_XGBoost.ipynb
│   ├── 10_Hyperparameter_Tuning_and_Optimization.ipynb
│   ├── 11_Model_Evaluation_and_Comparison.ipynb
│   └── Final_preprocessing_complete_steps.ipynb
│
├── report/                               # 📄 Official Project Documentation
│   ├── Anti_Money_Laundering_Detection_Project_Report.docx   # Word Report
│   └── Anti_Money_Laundering_Detection_Project_Report.pdf    # PDF Report
│
├── result/                               # 📈 Publication-Grade Visualizations
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── eda_distributions.png
│   └── correlation_matrix.png
│
├── src/                                  # Modular Source Code
│   ├── export_models.py                  # Model Training & Artifact Serialization
│   ├── inference.py                      # Production AMLDetector Engine
│   └── run_complete_pipeline.py          # Master CLI Automation Script
│
└── templates/                            # Web Application Frontend (HTML5/Bootstrap)
    ├── base.html
    ├── index.html
    ├── screener.html
    ├── batch.html
    ├── customers.html
    ├── models.html
    └── about.html
```

---

## ⚡ How to Run the Project

### Option A: Launch the Web Application (Easiest & Best for Viva Presentation)
Simply double-click `run_app.bat`!
Or run from terminal:
```bash
.venv\Scripts\python.exe app.py
```
Open **http://127.0.0.1:5000** in your browser.

### Option B: Open the Jupyter Notebooks
1. Launch Jupyter Notebook:
```bash
.venv\Scripts\jupyter.exe notebook notebook/AML_Detection_Complete_Project.ipynb
```
2. Select the Python environment at `.venv\Scripts\python.exe`.
3. All cells are already pre-executed with output, or you can click **Kernel -> Restart & Run All**.

---

## 📑 10 Instructor Guidelines Mapping

| Instructor Requirement | Covered In Notebook | Covered In Web App & Reports |
| :--- | :--- | :--- |
| **1. Dataset Description** | Section 1 in Master Notebook | Dashboard & About Page |
| **2. Preprocessing Results** | Section 2 (Zero Leakage) | Model Metadata & Report |
| **3. Exploratory Data Analysis** | Section 3 (All Figures) | Model Analytics Tab |
| **4. Feature Engineering** | Section 4 (15 Features) | Screener Explanations |
| **5. Model Building & Training** | Section 5 (6 Models, 5-Fold CV) | Model Analytics Hub |
| **6. Hyperparameter Tuning** | Section 6 (GridSearchCV) | Report & Notebook |
| **7. Model Evaluation** | Section 7 (Confusion Matrices) | Metrics Tab & Report |
| **8. Model Comparison** | Section 8 (Benchmark Table) | Models Page Table |
| **9. Results & Discussion** | Section 9 (In-depth inferences) | Report Document |
| **10. Final Prediction / Demo** | Section 10 (Live Demo) | Live Screener & Batch Tool |
