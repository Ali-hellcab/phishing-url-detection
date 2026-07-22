import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

ASSET_DIR = "deployment_assets"

# Load model, scaler, and columns on startup
model = tf.keras.models.load_model(os.path.join(ASSET_DIR, "cnn_lstm_model.keras"))
scaler = joblib.load(os.path.join(ASSET_DIR, "scaler.joblib"))

with open(os.path.join(ASSET_DIR, "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

def extract_url_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc if parsed_url.netloc else parsed_url.path
    domain_parts = domain.split('.')
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""
    no_of_subdomains = max(0, len(domain_parts) - 2)

    url_len = max(len(url), 1)
    no_of_letters = sum(c.isalpha() for c in url)
    no_of_digits = sum(c.isdigit() for c in url)
    no_of_spec_chars = sum(not c.isalnum() and c not in ['/', ':', '.', '-', '_', '?', '=', '&'] for c in url)

    obfuscated_chars = len(re.findall(r'%[0-9a-fA-F]{2}', url))

    return {
        'URLLength': url_len,
        'DomainLength': len(domain),
        'IsDomainIP': 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0,
        'TLDLength': len(tld),
        'NoOfSubDomain': no_of_subdomains,
        'HasObfuscation': 1 if obfuscated_chars > 0 else 0,
        'NoOfObfuscatedChar': obfuscated_chars,
        'ObfuscationRatio': obfuscated_chars / url_len,
        'NoOfLettersInURL': no_of_letters,
        'LetterRatioInURL': no_of_letters / url_len,
        'NoOfDegitsInURL': no_of_digits,
        'DegitRatioInURL': no_of_digits / url_len,
        'NoOfEqualsInURL': url.count('='),
        'NoOfQMarkInURL': url.count('?'),
        'NoOfAmpersandInURL': url.count('&'),
        'NoOfOtherSpecialCharsInURL': no_of_spec_chars,
        'SpacialCharRatioInURL': no_of_spec_chars / url_len,
        'IsHTTPS': 1 if parsed_url.scheme == 'https' else 0
    }

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API active", "model": "CNN-BiLSTM Phishing URL Classifier"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' key in JSON payload"}), 400

    raw_url = str(data['url']).strip()
    if not raw_url:
        return jsonify({"error": "Empty URL"}), 400

    # Extract 18 features and scale
    features = extract_url_features(raw_url)
    df = pd.DataFrame([features])[feature_columns]
    scaled = scaler.transform(df)
    cnn_input = np.expand_dims(scaled, axis=-1)

    prob = float(model.predict(cnn_input, verbose=0)[0][0])
    
    # Probability mapping
    is_legit = prob >= 0.5
    result = "Legitimate" if is_legit else "Phishing"

    return jsonify({
        "url": raw_url,
        "prediction": result,
        "confidence": {
            "legitimate": f"{(prob * 100):.2f}%",
            "phishing": f"{((1.0 - prob) * 100):.2f}%"
        },
        "raw_score": prob
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
