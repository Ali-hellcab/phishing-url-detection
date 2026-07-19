import json
import re
import numpy as np
import pandas as pd
import gradio as gr

from urllib.parse import urlparse
from tensorflow.keras.models import load_model
import joblib

# Load deployment assets
model = load_model("deployment_assets/cnn_lstm_model.keras")
scaler = joblib.load("deployment_assets/scaler.joblib")

with open("deployment_assets/feature_columns.json") as f:
    feature_columns = json.load(f)


def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc if parsed.netloc else parsed.path

    features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", domain) else 0,
        "NoOfSubDomain": domain.count("."),
        "NoOfDots": url.count("."),
        "NoOfHyphen": url.count("-"),
        "NoOfSlash": url.count("/"),
        "NoOfQuestionMark": url.count("?"),
        "NoOfEqual": url.count("="),
        "NoOfAt": url.count("@"),
        "NoOfAmpersand": url.count("&"),
        "NoOfDigits": sum(c.isdigit() for c in url),
        "NoOfLetters": sum(c.isalpha() for c in url),
    }

    df = pd.DataFrame([features])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df


def predict(url):
    df = extract_features(url)

    X = scaler.transform(df)

    X = np.expand_dims(X, axis=-1)

    probability = model.predict(X, verbose=0)[0][0]

    if probability >= 0.5:
        return f"⚠️ Phishing URL\nConfidence: {probability:.2%}"
    else:
        return f"✅ Benign URL\nConfidence: {(1-probability):.2%}"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, label="Enter URL"),
    outputs=gr.Textbox(label="Prediction"),
    title="Phishing URL Detection",
    description="CNN-BiLSTM based phishing URL detector."
)

demo.launch()