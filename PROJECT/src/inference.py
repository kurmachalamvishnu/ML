import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = r"C:\KL\4th Semister\ML\Anti-Money laundering project"
MODELS_DIR = os.path.join(BASE_DIR, "models")

class AMLDetector:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, "best_aml_model.joblib"))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        
        with open(os.path.join(MODELS_DIR, "customer_profiles.json"), "r", encoding="utf-8") as f:
            self.customer_profiles = json.load(f)
            
        with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        with open(os.path.join(MODELS_DIR, "metrics.json"), "r", encoding="utf-8") as f:
            self.metrics = json.load(f)
            
        self.feature_cols = self.metadata['features']

    def get_customer_profile(self, customer_id):
        cid_str = str(customer_id)
        if cid_str in self.customer_profiles:
            return self.customer_profiles[cid_str]
        # Default fallback for new/unknown customers
        return {
            'cust_mean_amount': 5000.0,
            'cust_std_amount': 2888.0,
            'cust_tx_count': 100
        }

    def predict_single(self, amount, customer_id, tx_type, date_str=None, rolling_7d=None):
        amount = float(amount)
        customer_id = int(customer_id)
        tx_type = tx_type.lower().strip()
        
        if not date_str:
            dt = pd.Timestamp.now()
        else:
            try:
                dt = pd.to_datetime(date_str)
            except:
                dt = pd.Timestamp.now()

        month = dt.month
        day = dt.day
        dayofweek = dt.dayofweek
        is_weekend = 1 if dayofweek in [5, 6] else 0

        # Fetch customer baseline
        profile = self.get_customer_profile(customer_id)
        c_mean = profile['cust_mean_amount']
        c_std = profile['cust_std_amount']
        c_count = profile['cust_tx_count']

        amount_to_cust_mean = amount / (c_mean + 1e-5)
        amount_zscore = (amount - c_mean) / (c_std + 1e-5)
        
        if rolling_7d is None:
            rolling_7d = 2 if amount < 5000 else 8
        rolling_7d = float(rolling_7d)

        structuring_flag = 1 if (amount >= 8500 and amount < 10000) else 0

        type_credit = 1 if tx_type == 'credit' else 0
        type_debit = 1 if tx_type == 'debit' else 0
        type_transfer = 1 if tx_type == 'transfer' else 0

        # Construct feature vector
        feat_dict = {
            'amount': amount,
            'cust_mean_amount': c_mean,
            'cust_std_amount': c_std,
            'cust_tx_count': c_count,
            'amount_to_cust_mean': amount_to_cust_mean,
            'amount_zscore': amount_zscore,
            'month': month,
            'day': day,
            'dayofweek': dayofweek,
            'is_weekend': is_weekend,
            'rolling_tx_7d': rolling_7d,
            'structuring_flag': structuring_flag,
            'type_credit': type_credit,
            'type_debit': type_debit,
            'type_transfer': type_transfer
        }

        X_df = pd.DataFrame([feat_dict])[self.feature_cols]
        X_scaled = self.scaler.transform(X_df)

        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        pred = int(self.model.predict(X_scaled)[0])

        # Generate AML Rule Explanations
        reasons = []
        if structuring_flag == 1:
            reasons.append(f"Structuring / Smurfing: Amount ${amount:,.2f} is intentionally just below the $10,000 regulatory reporting threshold.")
        if amount_zscore > 2.5:
            reasons.append(f"Customer Outlier: Amount is {amount_zscore:.1f} standard deviations above customer's historical average (${c_mean:,.2f}).")
        if rolling_7d > 10:
            reasons.append(f"Velocity Spike: Customer made {int(rolling_7d)} transactions within a 7-day window, indicating rapid movement of funds.")
        if tx_type == 'transfer' and amount > 7000:
            reasons.append(f"High-Value Layering: Large transfer amount typical of money layering phase.")
        if len(reasons) == 0:
            reasons.append("Transaction metrics align with standard customer baseline activity.")

        # Risk Tiering
        if proba >= 0.80:
            risk_tier = "CRITICAL RISK"
            badge_color = "danger"
        elif proba >= 0.50:
            risk_tier = "HIGH RISK"
            badge_color = "warning"
        elif proba >= 0.20:
            risk_tier = "MEDIUM RISK"
            badge_color = "info"
        else:
            risk_tier = "LOW RISK"
            badge_color = "success"

        return {
            'is_laundering': pred,
            'risk_score': proba,
            'risk_percentage': f"{proba * 100:.2f}%",
            'risk_tier': risk_tier,
            'badge_color': badge_color,
            'reasons': reasons,
            'input_summary': {
                'customer_id': customer_id,
                'amount': amount,
                'type': tx_type,
                'cust_mean': c_mean,
                'amount_zscore': amount_zscore,
                'rolling_7d': rolling_7d,
                'structuring_flag': bool(structuring_flag)
            }
        }

    def predict_batch_df(self, df):
        results = []
        for idx, row in df.iterrows():
            amt = row.get('amount', 100.0)
            cid = row.get('customer_id', 1)
            ttype = row.get('type', 'transfer')
            dt = row.get('date', None)
            res = self.predict_single(amt, cid, ttype, dt)
            results.append({
                'transaction_id': row.get('transaction_id', idx + 1),
                'customer_id': cid,
                'amount': amt,
                'type': ttype,
                'risk_score': res['risk_percentage'],
                'risk_tier': res['risk_tier'],
                'decision': "FLAGGED / LAUNDERING" if res['is_laundering'] == 1 else "APPROVED",
                'top_reason': res['reasons'][0]
            })
        return pd.DataFrame(results)

if __name__ == '__main__':
    detector = AMLDetector()
    print("Testing detector on normal sample:")
    res_normal = detector.predict_single(150.0, 10, 'credit')
    print("Normal Result:", res_normal['risk_tier'], res_normal['risk_percentage'])
    
    print("\nTesting detector on structuring sample:")
    res_laundering = detector.predict_single(9850.0, 10, 'transfer', rolling_7d=15)
    print("Laundering Result:", res_laundering['risk_tier'], res_laundering['risk_percentage'])
    print("Reasons:", res_laundering['reasons'])
