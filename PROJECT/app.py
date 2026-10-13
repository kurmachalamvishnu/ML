import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from src.inference import AMLDetector

BASE_DIR = r"C:\KL\4th Semister\ML\Anti-Money laundering project"
RESULT_DIR = os.path.join(BASE_DIR, "result")
REPORT_DIR = os.path.join(BASE_DIR, "report")
DATA_PATH = os.path.join(BASE_DIR, "data", "financial_transactions.csv")

app = Flask(__name__)
detector = AMLDetector()

@app.route('/')
def index():
    return render_template('index.html', active_page='dashboard')

@app.route('/screener')
def screener():
    return render_template('screener.html', active_page='screener')

@app.route('/batch')
def batch():
    return render_template('batch.html', active_page='batch')

@app.route('/customers')
def customers():
    return render_template('customers.html', active_page='customers')

@app.route('/models')
def models():
    return render_template('models.html', active_page='models')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

# Static plot serving
@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)

# Report downloads
@app.route('/download/pdf')
def download_pdf():
    pdf_path = os.path.join(REPORT_DIR, "Anti_Money_Laundering_Detection_Project_Report.pdf")
    return send_file(pdf_path, as_attachment=True)

@app.route('/download/docx')
def download_docx():
    docx_path = os.path.join(REPORT_DIR, "Anti_Money_Laundering_Detection_Project_Report.docx")
    return send_file(docx_path, as_attachment=True)

# APIs
@app.route('/api/screen-transaction', methods=['POST'])
def api_screen():
    data = request.get_json()
    amount = float(data.get('amount', 100.0))
    customer_id = int(data.get('customer_id', 1))
    tx_type = data.get('type', 'debit')
    rolling_7d = data.get('rolling_7d', None)
    date_str = data.get('date', None)

    res = detector.predict_single(amount, customer_id, tx_type, date_str, rolling_7d)
    return jsonify(res)

@app.route('/api/customer-profile/<int:cid>', methods=['GET'])
def api_profile(cid):
    profile = detector.get_customer_profile(cid)
    return jsonify(profile)

@app.route('/api/batch-sample', methods=['GET'])
def api_batch_sample():
    # Build 30 diverse transactions including normal, structuring, high velocity, etc.
    df_raw = pd.read_csv(DATA_PATH)
    sample_df = df_raw.sample(26, random_state=42).copy()
    
    # Inject 4 known suspicious patterns
    suspicious_samples = pd.DataFrame([
        {'transaction_id': 999901, 'customer_id': 191, 'amount': 9831.27, 'type': 'debit', 'date': '2026-09-10', 'description': 'Structuring transfer'},
        {'transaction_id': 999902, 'customer_id': 385, 'amount': 9458.75, 'type': 'transfer', 'date': '2026-09-11', 'description': 'Offshore wire'},
        {'transaction_id': 999903, 'customer_id': 287, 'amount': 9719.83, 'type': 'debit', 'date': '2026-09-12', 'description': 'Rapid cash debit'},
        {'transaction_id': 999904, 'customer_id': 433, 'amount': 9929.50, 'type': 'debit', 'date': '2026-09-13', 'description': 'Layered fund transfer'}
    ])

    batch_df = pd.concat([suspicious_samples, sample_df], ignore_index=True)
    results = detector.predict_batch_df(batch_df)
    return jsonify(results.to_dict(orient='records'))

@app.route('/api/batch-upload', methods=['POST'])
def api_batch_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file)
        results = detector.predict_batch_df(df)
        return jsonify(results.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting AML Sentinel Web Application on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
