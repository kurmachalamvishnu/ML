# Placement Prediction System 🎓💼

A Machine Learning project for predicting student placement status, analyzing key performance metrics, and evaluating multiple classification and regression algorithms.

## 📌 Project Overview
This repository contains end-to-end Machine Learning workflows designed to analyze student academic records, coding evaluations, soft skills, and internship experiences to predict placement outcomes.

## 📂 Project Structure
```text
PLACEMENT_PREDICTION SYSTEM/
├── data/                    # Processed & raw datasets
├── notebook/                # Jupyter Notebooks detailing EDA and model experiments
│   ├── 01_Dataset_Exploration.ipynb
│   ├── 02_EDA.ipynb
│   ├── Final_preprocessing_complete_steps.ipynb
│   ├── P3_SLR.ipynb & P3_MLR.ipynb & P3_GD.ipynb
│   ├── P4_1.Binary_Logistic_Regression_PlacementStatus.ipynb
│   ├── P4_2.Multinomial_Logistic_Regression_CGPA_Tier.ipynb
│   ├── P5_regularisation_Ridge_Lasso_ElasticNet.ipynb
│   ├── P7_Random_Forest_OOB_Bagging_Feature_Subsampling.ipynb
│   ├── P8_Boosting.ipynb
│   └── P9_K-means.ipynb
├── outputs/                 # Plots, correlation heatmaps, and boxplots
├── requirement.txt          # Python dependencies
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kurmachalamvishnu/PlacementPredict.git
cd PlacementPredict
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

### 4. Run the Jupyter Notebooks
```bash
jupyter notebook
```

## 🛠️ Tech Stack & Models
- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Algorithms:**
  - Simple & Multiple Linear Regression
  - Logistic Regression (Binary & Multinomial)
  - Regularization (Ridge, Lasso, ElasticNet)
  - Random Forest & Ensemble Learning
  - Boosting Algorithms
  - K-Means Clustering
