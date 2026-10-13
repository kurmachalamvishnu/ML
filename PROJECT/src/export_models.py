import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = r"C:\KL\4th Semister\ML\Anti-Money laundering project"
DATA_PATH = os.path.join(BASE_DIR, "data", "financial_transactions.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def export_all():
    print("Exporting production AML models and metadata...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['customer_id', 'date']).reset_index(drop=True)

    # 1. Compute and save customer profiles
    cust_stats = df.groupby('customer_id')['amount'].agg(
        cust_mean_amount='mean',
        cust_std_amount='std',
        cust_tx_count='count'
    ).reset_index()
    cust_stats['cust_std_amount'] = cust_stats['cust_std_amount'].fillna(0)

    # Export customer profiles to JSON for fast lookup
    profiles = {}
    for _, row in cust_stats.iterrows():
        cid = int(row['customer_id'])
        profiles[cid] = {
            'cust_mean_amount': float(row['cust_mean_amount']),
            'cust_std_amount': float(row['cust_std_amount']),
            'cust_tx_count': int(row['cust_tx_count'])
        }
    with open(os.path.join(MODELS_DIR, "customer_profiles.json"), "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print(f"Exported customer profiles for {len(profiles)} accounts.")

    # 2. Merge stats and generate ground truth target
    df = df.merge(cust_stats, on='customer_id', how='left')
    df['amount_to_cust_mean'] = df['amount'] / (df['cust_mean_amount'] + 1e-5)
    df['amount_zscore'] = (df['amount'] - df['cust_mean_amount']) / (df['cust_std_amount'] + 1e-5)

    structuring_mask = (df['amount'] >= 8500) & (df['amount'] < 10000) & (df['type'].isin(['transfer', 'debit']))
    zscore_mask = (df['amount_zscore'] > 2.7) & (df['amount'] > 7500)

    iso = IsolationForest(contamination=0.035, random_state=42)
    iso_preds = iso.fit_predict(df[['amount', 'amount_to_cust_mean', 'amount_zscore', 'cust_tx_count']])
    iso_mask = (iso_preds == -1)

    df['is_laundering'] = ((structuring_mask & iso_mask) | (zscore_mask & iso_mask) | (structuring_mask & (df['amount_zscore'] > 2.0))).astype(int)

    # 3. Features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    df.set_index('date', inplace=True)
    df['rolling_tx_7d'] = df.groupby('customer_id')['transaction_id'].transform(lambda x: x.rolling('7D').count())
    df.reset_index(inplace=True)

    df['structuring_flag'] = ((df['amount'] >= 8500) & (df['amount'] < 10000)).astype(int)
    type_dummies = pd.get_dummies(df['type'], prefix='type', drop_first=False).astype(int)
    df = pd.concat([df, type_dummies], axis=1)

    feature_cols = [
        'amount', 'cust_mean_amount', 'cust_std_amount', 'cust_tx_count',
        'amount_to_cust_mean', 'amount_zscore', 'month', 'day', 'dayofweek',
        'is_weekend', 'rolling_tx_7d', 'structuring_flag',
        'type_credit', 'type_debit', 'type_transfer'
    ]

    X = df[feature_cols]
    y = df['is_laundering']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("Saved scaler.joblib")

    # Train and save best model
    best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    best_model.fit(X_train_scaled, y_train)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_aml_model.joblib"))
    print("Saved best_aml_model.joblib")

    # Evaluate
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'model_name': 'Gradient Boosting Classifier',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'feature_names': feature_cols
    }

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        'features': feature_cols,
        'types': ['credit', 'debit', 'transfer'],
        'thresholds': {
            'structuring_min': 8500,
            'structuring_max': 9999,
            'zscore_high_risk': 2.7,
            'risk_decision_threshold': 0.50
        }
    }
    with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model metrics and metadata exported successfully!")
    print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")

if __name__ == '__main__':
    export_all()
