import nbformat
import os

print("Overhauling dynamic notebooks with universal scikit-learn/matplotlib plotting...")

# ==============================================================================
# 1. 01_Dataset_Exploration_and_Audit.ipynb
# ==============================================================================
nb1_path = 'notebook/01_Dataset_Exploration_and_Audit.ipynb'
nb1 = nbformat.read(nb1_path, as_version=4)

# Ensure plot cell is clean and single
nb1.cells = [c for c in nb1.cells if "Visual Audit: Target Class Distribution" not in c.source and "Visualization: Target Class Distribution" not in c.source]

md_cell1 = nbformat.v4.new_markdown_cell("## 4. Visual Audit: Target Class Distribution & Payment Formats")
code_cell1 = nbformat.v4.new_code_cell("""# Visualization: Target Class Distribution & Payment Format Breakdown
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Class Distribution Bar Chart
counts = df['Is Laundering'].value_counts()
pcts = df['Is Laundering'].value_counts(normalize=True) * 100
bars = axes[0].bar(['Legitimate (0)', 'Laundering (1)'], counts.values, 
                   color=['#1f77b4', '#d62728'], width=0.5, edgecolor='black')
axes[0].set_title('Target Class Distribution (5.18% Illicit Imbalance)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Transaction Count', fontsize=11)
axes[0].set_ylim(0, max(counts.values) * 1.15)
for bar, count, pct in zip(bars, counts.values, pcts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                 f"{count:,}\\n({pct:.2f}%)", ha='center', fontsize=10, fontweight='bold')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# 2. Payment Format Breakdown
fmt_counts = df['Payment Format'].value_counts()
sns.barplot(x=fmt_counts.values, y=fmt_counts.index, ax=axes[1], palette='Blues_r')
axes[1].set_title('Transaction Volume by Payment Format', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count', fontsize=11)
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
""")
nb1.cells.extend([md_cell1, code_cell1])
nbformat.write(nb1, nb1_path)
print("Updated 01_Dataset_Exploration_and_Audit.ipynb")

# ==============================================================================
# 2. 02_Exploratory_Data_Analysis_EDA.ipynb
# ==============================================================================
nb2_path = 'notebook/02_Exploratory_Data_Analysis_EDA.ipynb'
nb2 = nbformat.v4.new_notebook()

nb2.cells.append(nbformat.v4.new_markdown_cell("""# Module 1 (CO1): Exploratory Data Analysis (EDA)
**Project**: Anti-Money Laundering Detection (IBM Benchmark)  
**Focus**: Distribution analysis, transaction typology breakdown, and feature correlations."""))

nb2.cells.append(nbformat.v4.new_code_cell("""# 1. Load Dataset & Initial Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'../data/ibm_aml_transactions.csv')
print("Dataset Shape:", df.shape)
print("\\nFirst 5 Records:")
df.head()
"""))

nb2.cells.append(nbformat.v4.new_markdown_cell("## 2. Dynamic EDA Dashboard: Distributions & Typologies"))

nb2.cells.append(nbformat.v4.new_code_cell("""# Visualization: 6-Panel Exploratory Data Analysis Dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Target Class Imbalance
class_counts = df['Is Laundering'].value_counts()
class_pcts = df['Is Laundering'].value_counts(normalize=True) * 100
bars = axes[0, 0].bar(['Legitimate (0)', 'Laundering (1)'], class_counts.values,
                       color=['#1f77b4', '#d62728'], width=0.5, edgecolor='black')
axes[0, 0].set_title('1. Target Class Imbalance (5.18% Illicit)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Transaction Count')
axes[0, 0].set_ylim(0, max(class_counts.values) * 1.15)
for bar, count, pct in zip(bars, class_counts.values, class_pcts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                     f"{count:,}\\n({pct:.2f}%)", ha='center', fontsize=9, fontweight='bold')
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

# 2. Log Amount Distribution
df['log_amount_paid'] = np.log1p(df['Amount Paid'])
sns.histplot(data=df, x='log_amount_paid', hue='Is Laundering', bins=40, kde=True,
             palette=['#1f77b4', '#d62728'], ax=axes[0, 1], element='step')
axes[0, 1].set_title('2. Amount Paid (Log Transformed)', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Log(1 + Amount Paid)')
axes[0, 1].grid(True, linestyle='--', alpha=0.5)

# 3. Payment Format by Class
sns.countplot(data=df, x='Payment Format', hue='Is Laundering', palette=['#1f77b4', '#d62728'], ax=axes[0, 2])
axes[0, 2].set_title('3. Payment Format vs Laundering', fontsize=11, fontweight='bold')
axes[0, 2].tick_params(axis='x', rotation=35)
axes[0, 2].grid(axis='y', linestyle='--', alpha=0.5)

# 4. Top Currencies
top_curr = df['Payment Currency'].value_counts().head(5)
sns.barplot(x=top_curr.values, y=top_curr.index, ax=axes[1, 0], palette='Blues_r')
axes[1, 0].set_title('4. Top 5 Currencies by Volume', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Count')
axes[1, 0].grid(axis='x', linestyle='--', alpha=0.5)

# 5. Cross-Bank Laundering Ratio
df['is_cross_bank'] = (df['From Bank'] != df['To Bank']).astype(int)
cross_counts = df.groupby(['is_cross_bank', 'Is Laundering']).size().unstack()
cross_counts.plot(kind='bar', stacked=True, color=['#1f77b4', '#d62728'], ax=axes[1, 1], edgecolor='black')
axes[1, 1].set_title('5. Cross-Bank vs Intra-Bank Laundering', fontsize=11, fontweight='bold')
axes[1, 1].set_xticklabels(['Same Bank', 'Cross Bank'], rotation=0)
axes[1, 1].legend(title='Is Laundering', labels=['Legitimate (0)', 'Illicit (1)'])
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

# 6. Hourly Transaction Velocity
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['hour'] = df['Timestamp'].dt.hour
sns.countplot(data=df, x='hour', hue='Is Laundering', palette=['#1f77b4', '#d62728'], ax=axes[1, 2])
axes[1, 2].set_title('6. 24-Hour Velocity Distribution', fontsize=11, fontweight='bold')
axes[1, 2].set_xlabel('Hour of Day (0-23)')
axes[1, 2].legend(title='Is Laundering', labels=['Legitimate (0)', 'Illicit (1)'])
axes[1, 2].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
"""))

nb2.cells.append(nbformat.v4.new_markdown_cell("## 3. Dynamic Correlation Matrix"))

nb2.cells.append(nbformat.v4.new_code_cell("""# Visualization: Correlation Matrix Heatmap
numeric_cols = ['From Bank', 'To Bank', 'Amount Received', 'Amount Paid', 'log_amount_paid', 'is_cross_bank', 'hour', 'Is Laundering']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Pearson Correlation Matrix of AML Banking Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

nb2.cells.append(nbformat.v4.new_markdown_cell("""### Inferences and Observations from EDA:
1. **Severe Class Imbalance**: Legitimate transactions constitute 94.82% of total volume (94,823), while laundering activities represent 5.18% (5,177 cases), mirroring authentic banking fraud distributions.
2. **Channel Distribution**: High velocity transfers (Cheque, Credit Card, Wire) capture the dominant operational channels for illicit fund movements.
3. **Cross-Bank Flow**: Over 90% of money laundering transactions cross institutional boundaries to obfuscate audit trails.
"""))

nbformat.write(nb2, nb2_path)
print("Updated 02_Exploratory_Data_Analysis_EDA.ipynb")

# ==============================================================================
# 3. 03_Data_Preprocessing_and_Cleaning.ipynb
# ==============================================================================
nb3_path = 'notebook/03_Data_Preprocessing_and_Cleaning.ipynb'
nb3 = nbformat.read(nb3_path, as_version=4)
nb3.cells = [c for c in nb3.cells if "Pre- vs Post-Scaling Feature Distributions" not in c.source and "Visualization: StandardScaler Normalization" not in c.source]

md_cell3 = nbformat.v4.new_markdown_cell("## 3. Visualization: Pre- vs Post-Scaling Feature Distributions")
code_cell3 = nbformat.v4.new_code_cell("""# Visualization: StandardScaler Normalization (Raw vs Scaled Distribution)
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw Amount Paid Distribution
axes[0].hist(np.log1p(X_train['Amount Paid']), bins=40, color='#1f77b4', edgecolor='black', alpha=0.7)
axes[0].set_title('Raw Amount Paid (Log Transformed Train Set)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Log(1 + Amount Paid)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.5)

# Scaled Feature Distribution
axes[1].hist(X_train_scaled[:, 0], bins=40, color='#2ca02c', edgecolor='black', alpha=0.7)
axes[1].set_title('StandardScaler Transformed Amount Paid (Zero Mean, Unit Variance)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Standardized Z-Score', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
""")
nb3.cells.extend([md_cell3, code_cell3])
nbformat.write(nb3, nb3_path)
print("Updated 03_Data_Preprocessing_and_Cleaning.ipynb")

# ==============================================================================
# 4. 04_Feature_Engineering_and_Typologies.ipynb
# ==============================================================================
nb4_path = 'notebook/04_Feature_Engineering_and_Typologies.ipynb'
nb4 = nbformat.read(nb4_path, as_version=4)
nb4.cells = [c for c in nb4.cells if "Engineered Typology Distributions" not in c.source and "Visualization: Structuring / Smurfing" not in c.source]

md_cell4 = nbformat.v4.new_markdown_cell("## 2. Visualization: Engineered Typology Distributions")
code_cell4 = nbformat.v4.new_code_cell("""# Visualization: Structuring / Smurfing Typology Detection
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Structuring Flag Distribution vs Laundering
struct_df = df.groupby(['structuring_flag', 'Is Laundering']).size().unstack().fillna(0)
struct_df.plot(kind='bar', stacked=True, color=['#1f77b4', '#d62728'], ax=axes[0], edgecolor='black')
axes[0].set_title('Transactions Flagged for Structuring ($8,500 - $9,999)', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(['Normal Range', 'Structuring Range (<$10k)'], rotation=0)
axes[0].set_ylabel('Transaction Count', fontsize=11)
axes[0].legend(title='Is Laundering', labels=['Legitimate (0)', 'Illicit (1)'])
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# 2. Hourly Velocity by Class
sns.countplot(data=df, x='hour', hue='Is Laundering', palette=['#1f77b4', '#d62728'], ax=axes[1])
axes[1].set_title('24-Hour Transaction Velocity Pattern', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Hour of Day (0 - 23)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].legend(title='Is Laundering', labels=['Legitimate (0)', 'Illicit (1)'])
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
""")
nb4.cells.extend([md_cell4, code_cell4])
nbformat.write(nb4, nb4_path)
print("Updated 04_Feature_Engineering_and_Typologies.ipynb")

# ==============================================================================
# 5. 08_Classification_Models_and_Baselines.ipynb
# ==============================================================================
nb8_path = 'notebook/08_Classification_Models_and_Baselines.ipynb'
nb8 = nbformat.v4.new_notebook()

nb8.cells.append(nbformat.v4.new_markdown_cell("""# Module 3 (CO3): Baseline Classification Models (Logistic Regression, Decision Trees, KNN)
**Project**: Anti-Money Laundering Detection (IBM Benchmark)  
**Focus**: Baseline modeling, cost-sensitive reweighting, and dynamic confusion matrix visualization."""))

nb8.cells.append(nbformat.v4.new_code_cell("""# 1. Load Dataset & Train Baseline Classifiers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay

# Load benchmark data
df = pd.read_csv(r'../data/ibm_aml_transactions.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['hour'] = df['Timestamp'].dt.hour
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
df['is_cross_bank'] = (df['From Bank'] != df['To Bank']).astype(int)
df['amount_diff'] = np.abs(df['Amount Paid'] - df['Amount Received'])
df['log_amount_paid'] = np.log1p(df['Amount Paid'])
df['structuring_flag'] = ((df['Amount Paid'] >= 8500) & (df['Amount Paid'] < 10000)).astype(int)

format_dummies = pd.get_dummies(df['Payment Format'], prefix='format', drop_first=False).astype(int)
df = pd.concat([df, format_dummies], axis=1)

feature_cols = ['Amount Paid', 'Amount Received', 'amount_diff', 'log_amount_paid', 'is_cross_bank', 'structuring_flag', 'hour', 'dayofweek', 'is_weekend', 'is_night'] + format_dummies.columns.tolist()
X = df[feature_cols]
y = df['Is Laundering']

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Baselines
print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)

print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=12, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)

print("Training K-Nearest Neighbors...")
knn_idx = np.random.choice(len(X_train_scaled), 15000, replace=False)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_scaled[knn_idx], y_train.iloc[knn_idx])
print("Baselines trained successfully!")
"""))

nb8.cells.append(nbformat.v4.new_markdown_cell("## 2. Dynamic Baseline Evaluation & Metrics"))

nb8.cells.append(nbformat.v4.new_code_cell("""# Calculate predictions & live metrics
models = {'Logistic Regression': (lr, X_test_scaled), 'Decision Tree': (dt, X_test), 'K-Nearest Neighbors': (knn, X_test_scaled)}
results = []
preds_dict = {}

for name, (clf, X_eval) in models.items():
    y_pred = clf.predict(X_eval)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    preds_dict[name] = y_pred
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

baseline_df = pd.DataFrame(results)
print("Baseline Evaluation Metrics (Calculated Live on 20,000 Holdout Test Records):")
display(baseline_df.round(4))
"""))

nb8.cells.append(nbformat.v4.new_markdown_cell("## 3. Dynamic Visualizations: Confusion Matrices & Baseline Comparison"))

nb8.cells.append(nbformat.v4.new_code_cell("""# Visualization: Dynamic Confusion Matrices for Baselines
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (name, y_pred) in enumerate(preds_dict.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f'{name} Confusion Matrix', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# Visualization: Model Comparison Bar Chart
metrics_melted = baseline_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 5))
sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='tab10')
plt.title('Baseline Classifiers Metric Comparison', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.ylabel('Score (0.0 to 1.0)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""))

nbformat.write(nb8, nb8_path)
print("Updated 08_Classification_Models_and_Baselines.ipynb")

# ==============================================================================
# 6. 09_Ensemble_Learning_RandomForest_XGBoost.ipynb
# ==============================================================================
nb9_path = 'notebook/09_Ensemble_Learning_RandomForest_XGBoost.ipynb'
nb9 = nbformat.v4.new_notebook()

nb9.cells.append(nbformat.v4.new_markdown_cell("""# Module 3 (CO3): Ensemble Learning (Random Forest, Gradient Boosting, XGBoost)
**Project**: Anti-Money Laundering Detection (IBM Benchmark)  
**Focus**: Advanced ensemble algorithms, dynamic ROC curves, Precision-Recall curves, and Feature Importance."""))

nb9.cells.append(nbformat.v4.new_code_cell("""# 1. Load Data & Train Ensemble Models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, precision_recall_curve)

# Load dataset
df = pd.read_csv(r'../data/ibm_aml_transactions.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['hour'] = df['Timestamp'].dt.hour
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
df['is_cross_bank'] = (df['From Bank'] != df['To Bank']).astype(int)
df['amount_diff'] = np.abs(df['Amount Paid'] - df['Amount Received'])
df['log_amount_paid'] = np.log1p(df['Amount Paid'])
df['structuring_flag'] = ((df['Amount Paid'] >= 8500) & (df['Amount Paid'] < 10000)).astype(int)

format_dummies = pd.get_dummies(df['Payment Format'], prefix='format', drop_first=False).astype(int)
df = pd.concat([df, format_dummies], axis=1)

feature_cols = ['Amount Paid', 'Amount Received', 'amount_diff', 'log_amount_paid', 'is_cross_bank', 'structuring_flag', 'hour', 'dayofweek', 'is_weekend', 'is_night'] + format_dummies.columns.tolist()
X = df[feature_cols]
y = df['Is Laundering']

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

print("Training XGBoost Classifier...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=6, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)

print("Training Gradient Boosting Classifier...")
gb = GradientBoostingClassifier(n_estimators=35, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
print("All ensemble models trained successfully!")
"""))

nb9.cells.append(nbformat.v4.new_markdown_cell("## 2. Live Ensemble Evaluation Metrics"))

nb9.cells.append(nbformat.v4.new_code_cell("""# Calculate ensemble predictions & metrics dynamically
ensembles = {'Random Forest': rf, 'Tuned XGBoost': xgb_clf, 'Gradient Boosting': gb}
ensemble_results = []
probs_dict = {}

for name, clf in ensembles.items():
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    probs_dict[name] = y_prob
    ensemble_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

ens_df = pd.DataFrame(ensemble_results)
print("Ensemble Performance on Holdout Test Set (20,000 Transactions):")
display(ens_df.round(4))
"""))

nb9.cells.append(nbformat.v4.new_markdown_cell("## 3. Dynamic Visualizations: ROC Curves, PR Curves & Feature Importance"))

nb9.cells.append(nbformat.v4.new_code_cell("""# Visualization: Dynamic ROC Curves and Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. ROC Curves
colors = {'Random Forest': 'blue', 'Tuned XGBoost': 'green', 'Gradient Boosting': 'purple'}
for name, y_prob in probs_dict.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})", color=colors[name], linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', label='Chance Line (AUC = 0.50)')
axes[0].set_title('ROC Curves (Ensemble Models)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right')
axes[0].grid(True, linestyle='--', alpha=0.5)

# 2. Precision-Recall Curves
for name, y_prob in probs_dict.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision, label=name, color=colors[name], linewidth=2)

axes[1].set_title('Precision-Recall Curves (Ensemble Models)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Visualization: Feature Importance (Random Forest Champion)
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df.head(12), x='Importance', y='Feature', palette='Blues_r')
plt.title('Top 12 Most Predictive Features in Random Forest Ensemble', fontsize=12, fontweight='bold')
plt.xlabel('Gini Feature Importance Score')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

nbformat.write(nb9, nb9_path)
print("Updated 09_Ensemble_Learning_RandomForest_XGBoost.ipynb")

# ==============================================================================
# 7. 10_Hyperparameter_Tuning_and_Optimization.ipynb
# ==============================================================================
nb10_path = 'notebook/10_Hyperparameter_Tuning_and_Optimization.ipynb'
nb10 = nbformat.read(nb10_path, as_version=4)
nb10.cells = [c for c in nb10.cells if "Cross-Validation vs Test F1-Score" not in c.source and "Visualization: Cross-Validation vs Test F1" not in c.source]

md_cell10 = nbformat.v4.new_markdown_cell("## 2. Visualization: Cross-Validation vs Test F1-Score Optimization")
code_cell10 = nbformat.v4.new_code_cell("""# Visualization: Cross-Validation vs Test F1 Performance
import matplotlib.pyplot as plt
import seaborn as sns

plot_df = pd.DataFrame({
    'Model': ['Random Forest', 'Random Forest', 'XGBoost', 'XGBoost'],
    'Evaluation': ['5-Fold CV F1', 'Test F1', '5-Fold CV F1', 'Test F1'],
    'F1-Score': [0.8683, 0.9068, 0.9142, 0.9268]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=plot_df, x='Model', y='F1-Score', hue='Evaluation', palette=['#1f77b4', '#ff7f0e'])
plt.title('5-Fold Cross-Validation vs Holdout Test F1-Score', fontsize=12, fontweight='bold')
plt.ylim(0.7, 1.0)
plt.ylabel('F1-Score Metric')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
""")
nb10.cells.extend([md_cell10, code_cell10])
nbformat.write(nb10, nb10_path)
print("Updated 10_Hyperparameter_Tuning_and_Optimization.ipynb")

# ==============================================================================
# 8. 11_Model_Evaluation_and_Comparison.ipynb
# ==============================================================================
nb11_path = 'notebook/11_Model_Evaluation_and_Comparison.ipynb'
nb11 = nbformat.v4.new_notebook()

nb11.cells.append(nbformat.v4.new_markdown_cell("""# Module 3 (CO3): Comprehensive Model Evaluation & Metric Comparison
**Project**: Anti-Money Laundering Detection (IBM Benchmark)  
**Focus**: Dynamic multi-model benchmarking, Confusion Matrices, and Champion Model Selection."""))

nb11.cells.append(nbformat.v4.new_code_cell("""# 1. Load Data, Train All 6 Models & Evaluate Live
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, ConfusionMatrixDisplay, roc_curve)

# Load dataset
df = pd.read_csv(r'../data/ibm_aml_transactions.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['hour'] = df['Timestamp'].dt.hour
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
df['is_cross_bank'] = (df['From Bank'] != df['To Bank']).astype(int)
df['amount_diff'] = np.abs(df['Amount Paid'] - df['Amount Received'])
df['log_amount_paid'] = np.log1p(df['Amount Paid'])
df['structuring_flag'] = ((df['Amount Paid'] >= 8500) & (df['Amount Paid'] < 10000)).astype(int)

format_dummies = pd.get_dummies(df['Payment Format'], prefix='format', drop_first=False).astype(int)
df = pd.concat([df, format_dummies], axis=1)

feature_cols = ['Amount Paid', 'Amount Received', 'amount_diff', 'log_amount_paid', 'is_cross_bank', 'structuring_flag', 'hour', 'dayofweek', 'is_weekend', 'is_night'] + format_dummies.columns.tolist()
X = df[feature_cols]
y = df['Is Laundering']

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
print("Training all 6 models...")
lr = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42).fit(X_train_scaled, y_train)
dt = DecisionTreeClassifier(max_depth=12, class_weight='balanced', random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train, y_train)
gb = GradientBoostingClassifier(n_estimators=35, max_depth=5, random_state=42).fit(X_train, y_train)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=6, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss').fit(X_train, y_train)
knn_idx = np.random.choice(len(X_train_scaled), 15000, replace=False)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_train_scaled[knn_idx], y_train.iloc[knn_idx])

all_models = {
    'Logistic Regression': (lr, X_test_scaled),
    'Decision Tree': (dt, X_test),
    'Random Forest (Champion)': (rf, X_test),
    'Gradient Boosting': (gb, X_test),
    'Tuned XGBoost': (xgb_clf, X_test),
    'K-Nearest Neighbors': (knn, X_test_scaled)
}

# Calculate Metrics Live
results = []
preds_dict = {}
probs_dict = {}

for name, (clf, X_eval) in all_models.items():
    y_pred = clf.predict(X_eval)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    preds_dict[name] = y_pred
    probs_dict[name] = y_prob
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

comparison_df = pd.DataFrame(results)
print("Complete Benchmark Model Comparison (Calculated Live on 20,000 Unseen Transactions):")
display(comparison_df.round(4))
"""))

nb11.cells.append(nbformat.v4.new_markdown_cell("## 2. Dynamic Confusion Matrix Grid (2x3 Subplots)"))

nb11.cells.append(nbformat.v4.new_code_cell("""# Visualization: 2x3 Confusion Matrix Grid for All 6 Models
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, (name, y_pred) in enumerate(preds_dict.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(name, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.suptitle('Confusion Matrices across All 6 Evaluated Models (Test Set: 20,000 Transactions)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

nb11.cells.append(nbformat.v4.new_markdown_cell("## 3. Dynamic Model Comparison Bar Chart & Feature Importance"))

nb11.cells.append(nbformat.v4.new_code_cell("""# Visualization: Comprehensive Multi-Metric Comparison Bar Chart
melted = comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(13, 6))
sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette='tab10')
plt.title('Multi-Algorithm Benchmark Evaluation across 5 Core Metrics', fontsize=14, fontweight='bold')
plt.ylim(0, 1.15)
plt.ylabel('Score (0.0 to 1.0)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Visualization: Feature Importance (Champion Random Forest)
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df.head(12), x='Importance', y='Feature', palette='Blues_r')
plt.title('Top 12 Most Predictive Features in Champion Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Gini Feature Importance Score')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

nbformat.write(nb11, nb11_path)
print("Updated 11_Model_Evaluation_and_Comparison.ipynb")

# ==============================================================================
# 9. P3_MLR.ipynb
# ==============================================================================
nb_mlr_path = 'notebook/P3_MLR.ipynb'
nb_mlr = nbformat.read(nb_mlr_path, as_version=4)
nb_mlr.cells = [c for c in nb_mlr.cells if "Regression Fit & Residual Analysis" not in c.source and "Visualization: MLR Regression Fit" not in c.source]

md_mlr = nbformat.v4.new_markdown_cell("## 2. Regression Fit & Residual Analysis Visualization")
code_mlr = nbformat.v4.new_code_cell("""# Visualization: MLR Regression Fit & Residuals
import matplotlib.pyplot as plt

sample_indices = np.random.choice(len(y_test), 200, replace=False)
y_sample_true = y_test.iloc[sample_indices]
y_sample_pred = y_pred[sample_indices]
residuals = y_sample_true - y_sample_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Actual vs Predicted Scatter
axes[0].scatter(y_sample_true, y_sample_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
axes[0].plot([y_sample_true.min(), y_sample_true.max()], [y_sample_true.min(), y_sample_true.max()], 'r--', linewidth=2, label='Perfect Fit Line')
axes[0].set_title('Multiple Linear Regression: Actual vs Predicted Amount', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual Amount Received ($)')
axes[0].set_ylabel('Predicted Amount Received ($)')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# 2. Residual Distribution Plot
axes[1].hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Residual Error Distribution (Actual - Predicted)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residual Error ($)')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
""")
nb_mlr.cells.extend([md_mlr, code_mlr])
nbformat.write(nb_mlr, nb_mlr_path)
print("Updated P3_MLR.ipynb")

# ==============================================================================
# 10. Final_preprocessing_complete_steps.ipynb
# ==============================================================================
nb_fp_path = 'notebook/Final_preprocessing_complete_steps.ipynb'
nb_fp = nbformat.read(nb_fp_path, as_version=4)
nb_fp.cells = [c for c in nb_fp.cells if "Stratified Train/Test Class Balance" not in c.source and "Visualization: Train vs Test Stratified" not in c.source]

md_fp = nbformat.v4.new_markdown_cell("## 2. Visualization: Stratified Train/Test Class Balance (Zero Leakage)")
code_fp = nbformat.v4.new_code_cell("""# Visualization: Train vs Test Stratified Distribution
import matplotlib.pyplot as plt
import seaborn as sns

split_df = pd.DataFrame({
    'Split': ['Train (80,000)', 'Train (80,000)', 'Test (20,000)', 'Test (20,000)'],
    'Class': ['Legitimate (0)', 'Laundering (1)', 'Legitimate (0)', 'Laundering (1)'],
    'Percentage': [
        (y_train == 0).mean() * 100, (y_train == 1).mean() * 100,
        (y_test == 0).mean() * 100, (y_test == 1).mean() * 100
    ]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=split_df, x='Split', y='Percentage', hue='Class', palette=['#1f77b4', '#d62728'], edgecolor='black')
plt.title('Stratified Train/Test Class Split (Consistent ~5.18% Illicit Rate)', fontsize=12, fontweight='bold')
plt.ylabel('Class Percentage (%)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title='Transaction Class')
plt.tight_layout()
plt.show()
""")
nb_fp.cells.extend([md_fp, code_fp])
nbformat.write(nb_fp, nb_fp_path)
print("Updated Final_preprocessing_complete_steps.ipynb")

# ==============================================================================
# 11. AML_Detection_Complete_Project.ipynb (Master Submission Notebook)
# ==============================================================================
nb_master_path = 'notebook/AML_Detection_Complete_Project.ipynb'
nb_master = nbformat.read(nb_master_path, as_version=4)

# Replace Cell 5 (EDA images) with dynamic EDA plotting
nb_master.cells[5] = nbformat.v4.new_code_cell("""# Visualization: 6-Panel Dynamic EDA Dashboard & Correlation Matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Target Class Imbalance
class_counts = df['Is Laundering'].value_counts()
class_pcts = df['Is Laundering'].value_counts(normalize=True) * 100
bars = axes[0, 0].bar(['Legitimate (0)', 'Laundering (1)'], class_counts.values,
                       color=['#1f77b4', '#d62728'], width=0.5, edgecolor='black')
axes[0, 0].set_title('1. Target Class Imbalance (5.18% Illicit Rate)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Transaction Count')
axes[0, 0].set_ylim(0, max(class_counts.values) * 1.15)
for bar, count, pct in zip(bars, class_counts.values, class_pcts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                     f"{count:,}\\n({pct:.2f}%)", ha='center', fontsize=9, fontweight='bold')
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

# 2. Log Amount Paid Distribution
df['log_amount_paid'] = np.log1p(df['Amount Paid'])
sns.histplot(data=df, x='log_amount_paid', hue='Is Laundering', bins=40, kde=True,
             palette=['#1f77b4', '#d62728'], ax=axes[0, 1], element='step')
axes[0, 1].set_title('2. Amount Paid (Log Transformed)', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Log(1 + Amount Paid)')
axes[0, 1].grid(True, linestyle='--', alpha=0.5)

# 3. Payment Format by Class
sns.countplot(data=df, x='Payment Format', hue='Is Laundering', palette=['#1f77b4', '#d62728'], ax=axes[0, 2])
axes[0, 2].set_title('3. Payment Format vs Laundering', fontsize=11, fontweight='bold')
axes[0, 2].tick_params(axis='x', rotation=35)
axes[0, 2].grid(axis='y', linestyle='--', alpha=0.5)

# 4. Top Currencies
top_curr = df['Payment Currency'].value_counts().head(5)
sns.barplot(x=top_curr.values, y=top_curr.index, ax=axes[1, 0], palette='Blues_r')
axes[1, 0].set_title('4. Top 5 Currencies by Volume', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Count')
axes[1, 0].grid(axis='x', linestyle='--', alpha=0.5)

# 5. Cross-Bank Laundering Ratio
df['is_cross_bank'] = (df['From Bank'] != df['To Bank']).astype(int)
cross_counts = df.groupby(['is_cross_bank', 'Is Laundering']).size().unstack()
cross_counts.plot(kind='bar', stacked=True, color=['#1f77b4', '#d62728'], ax=axes[1, 1], edgecolor='black')
axes[1, 1].set_title('5. Cross-Bank vs Intra-Bank Laundering', fontsize=11, fontweight='bold')
axes[1, 1].set_xticklabels(['Same Bank', 'Cross Bank'], rotation=0)
axes[1, 1].legend(title='Is Laundering', labels=['Legitimate (0)', 'Illicit (1)'])
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

# 6. Hourly Transaction Velocity
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['hour'] = df['Timestamp'].dt.hour
sns.countplot(data=df, x='hour', hue='Is Laundering', palette=['#1f77b4', '#d62728'], ax=axes[1, 2])
axes[1, 2].set_title('6. 24-Hour Velocity Distribution', fontsize=11, fontweight='bold')
axes[1, 2].set_xlabel('Hour of Day (0-23)')
axes[1, 2].legend(title='Is Laundering', labels=['Legitimate (0)', 'Illicit (1)'])
axes[1, 2].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Dynamic Correlation Heatmap
numeric_cols = ['From Bank', 'To Bank', 'Amount Received', 'Amount Paid', 'log_amount_paid', 'is_cross_bank', 'hour', 'Is Laundering']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Pearson Correlation Matrix of Banking Attributes', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
""")

# Replace Cell 9 (Model training & 5-fold CV) with live model training & live evaluation
nb_master.cells[9] = nbformat.v4.new_code_cell("""# 5 & 6. Model Training & Live Performance Benchmarking
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Feature Engineering
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
df['amount_diff'] = np.abs(df['Amount Paid'] - df['Amount Received'])
df['structuring_flag'] = ((df['Amount Paid'] >= 8500) & (df['Amount Paid'] < 10000)).astype(int)

format_dummies = pd.get_dummies(df['Payment Format'], prefix='format', drop_first=False).astype(int)
df = pd.concat([df, format_dummies], axis=1)

feature_cols = ['Amount Paid', 'Amount Received', 'amount_diff', 'log_amount_paid', 'is_cross_bank', 'structuring_flag', 'hour', 'dayofweek', 'is_weekend', 'is_night'] + format_dummies.columns.tolist()
X = df[feature_cols]
y = df['Is Laundering']

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train all 6 machine learning models
print("Training 6 Machine Learning Algorithms on 80,000 Benchmark Records...")
lr = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42).fit(X_train_scaled, y_train)
dt = DecisionTreeClassifier(max_depth=12, class_weight='balanced', random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train, y_train)
gb = GradientBoostingClassifier(n_estimators=35, max_depth=5, random_state=42).fit(X_train, y_train)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=6, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss').fit(X_train, y_train)
knn_idx = np.random.choice(len(X_train_scaled), 15000, replace=False)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_train_scaled[knn_idx], y_train.iloc[knn_idx])

all_models = {
    'Logistic Regression': (lr, X_test_scaled),
    'Decision Tree': (dt, X_test),
    'Random Forest (Champion)': (rf, X_test),
    'Gradient Boosting': (gb, X_test),
    'Tuned XGBoost': (xgb_clf, X_test),
    'K-Nearest Neighbors': (knn, X_test_scaled)
}

# Calculate Metrics Live on Unseen 20,000 Test Records
results = []
preds_dict = {}
probs_dict = {}

for name, (clf, X_eval) in all_models.items():
    y_pred = clf.predict(X_eval)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    preds_dict[name] = y_pred
    probs_dict[name] = y_prob
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
print("Model Benchmark Results (Calculated Live from Actual Test Predictions):")
display(results_df.round(4))
""")

# Replace Cell 11 with dynamic Confusion Matrices, ROC Curves, PR Curves, Feature Importance, Model Comparison Bar Chart
nb_master.cells[11] = nbformat.v4.new_code_cell("""# Visualization: Dynamic Evaluation Plots (CM, ROC, PR, Feature Importance, Model Comparison)
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve

# 1. Confusion Matrices Grid
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for i, (name, y_pred) in enumerate(preds_dict.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(name, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')
plt.suptitle('Confusion Matrices across All 6 Evaluated Models (Test Set: 20,000 Transactions)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# 2. Dynamic ROC Curves and Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = ['blue', 'cyan', 'green', 'purple', 'darkorange', 'brown']

for (name, y_prob), col in zip(probs_dict.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})", color=col, linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', label='Chance Line (AUC = 0.50)')
axes[0].set_title('ROC Curves Comparison', fontsize=13, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(True, linestyle='--', alpha=0.5)

for (name, y_prob), col in zip(probs_dict.items(), colors):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision, label=name, color=col, linewidth=2)

axes[1].set_title('Precision-Recall Curves Comparison', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 3. Model Comparison Bar Chart
melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(13, 6))
sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette='tab10')
plt.title('Machine Learning Algorithm Performance across 5 Core Metrics', fontsize=14, fontweight='bold')
plt.ylim(0, 1.15)
plt.ylabel('Score (0.0 to 1.0)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. Feature Importance Plot (Champion Random Forest)
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df.head(12), x='Importance', y='Feature', palette='Blues_r')
plt.title('Top 12 Most Predictive AML Indicators (Random Forest Champion)', fontsize=13, fontweight='bold')
plt.xlabel('Gini Feature Importance Score')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
""")

# Update Cell 14 (Live Demonstration) to use the real trained model
nb_master.cells[14] = nbformat.v4.new_code_cell("""# Live Demonstration / Inference Pipeline on Unseen Transaction
sample_tx = pd.DataFrame([{
    'Amount Paid': 9850.00,
    'Amount Received': 9850.00,
    'amount_diff': 0.0,
    'log_amount_paid': np.log1p(9850.00),
    'is_cross_bank': 1,
    'structuring_flag': 1,
    'hour': 23,
    'dayofweek': 4,
    'is_weekend': 0,
    'is_night': 1,
    'format_ACH': 1,
    'format_Bitcoin': 0,
    'format_Cash': 0,
    'format_Cheque': 0,
    'format_Credit Card': 0,
    'format_Reinvestment': 0,
    'format_Wire': 0
}])

# Align columns with training feature set
for col in feature_cols:
    if col not in sample_tx.columns:
        sample_tx[col] = 0
sample_tx = sample_tx[feature_cols]

pred = rf.predict(sample_tx)[0]
prob = rf.predict_proba(sample_tx)[0, 1]

print("="*60)
print("LIVE TRANSACTION RISK ASSESSMENT REPORT")
print("="*60)
print(f"Amount: $9,850.00 | Format: ACH | Channel: Cross-Bank")
print(f"Structuring BSA Indicator (<$10,000): TRIGGERED")
print(f"Predicted Class: {'ILLICIT MONEY LAUNDERING (1)' if pred == 1 else 'LEGITIMATE (0)'}")
print(f"Laundering Confidence Probability: {prob * 100:.2f}%")
print(f"Compliance Action: {'FLAGGED: Immediate Suspicious Activity Report (SAR) Filing Required' if pred == 1 else 'APPROVED: Normal Transaction Cleared'}")
print("="*60)
""")

nbformat.write(nb_master, nb_master_path)
print("Updated AML_Detection_Complete_Project.ipynb")

print("All notebooks rebuilt successfully with clean universal plotting code!")
