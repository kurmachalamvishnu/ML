# Explainable Student Placement Prediction Using Hybrid Machine Learning

**Course:** Machine Learning — Bachelor of Technology (A.Y. 2026–27)  
**Department:** Department of Computer Science and Engineering  

---

## 👥 Project Team & Mentorship
* **Students:**
  * **K. Vishnu** (Roll No: 2520030199)
  * **U. Vidhath** (Roll No: 2520030597)
* **Guide:** **Nirmalajyothi Narisetty**, Department of Computer Science and Engineering

---

## 📖 Abstract
The rapid evolution of data-driven technologies has transformed the way educational institutions evaluate student employability and career readiness. This project focuses on designing a predictive system that not only forecasts student placement outcomes but also provides transparent insights into the factors influencing these predictions. 

Unlike traditional black-box models, the proposed framework integrates multiple machine learning techniques into a hybrid approach, balancing predictive accuracy with interpretability. The methodology combines ensemble learning models such as Random Forest and Gradient Boosting with interpretable algorithms like Logistic Regression and Decision Trees. 

### Key Features Evaluated:
* **Academic Performance:** CGPA, semester-wise SGPA, core coursework
* **Technical Skills:** Coding test scores, aptitude test performance, technical certifications, projects
* **Professional Readiness:** Internships, publications, workshops attended
* **Interpersonal & Extracurricular:** Soft skills ratings, extracurricular involvement, mock interview scores

### Explainability Framework:
To ensure transparency and trust in model predictions, explainability techniques such as **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations) are integrated, enabling stakeholders and academic advisors to understand the exact contribution of each factor toward placement outcomes.

---

## 📁 Repository Structure
```text
PLACEMENT PREDICTION/
├── ABSRACT ML.docx          # Official Project Report Abstract
├── README.md                # Project Overview & Details
├── requirement.txt          # Python dependencies
├── data/                    # Datasets (Raw & Preprocessed Train/Test splits)
├── notebook/                # Jupyter Notebooks (EDA, Preprocessing, Modeling & Evaluation)
└── outputs/                 # Correlation heatmaps, feature distribution & boxplots
```
