# 🛡️ UPI Scam SMS Detector

An NLP-powered app that detects fraudulent UPI SMS messages in **5 languages** — Tamil, Telugu, Hindi, English, and Mixed — with explainable AI, rule-based signal detection, and sender verification.

> Built as a portfolio project to demonstrate end-to-end ML engineering, multilingual NLP, and explainable AI.

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://upi-scam-sms-detector-ghgwrj6c5xdjatsqqmqsze.streamlit.app/)

---

## What It Does

Paste any UPI-related SMS → get an instant verdict:

| Signal | What it checks |
|--------|---------------|
| 🤖 ML Model | Logistic Regression on TF-IDF features — classifies content as Scam / Legit |
| 🚩 Red Flag Detector | 8 regex patterns — urgency, OTP requests, fake URLs, KYC threats, prize scams |
| 📨 Sender Verifier | 80+ known DLT sender codes — flags personal mobile numbers and unknown IDs |
| ⚖️ Combined Verdict | 6-case decision matrix merging all signals into one plain-English conclusion |
| 🔬 LIME Explanation | Highlights exactly which words drove the ML model's decision |
| 🛡️ Safety Tips | Actionable next steps for every Scam or Suspicious verdict |

---

## Tech Stack

| Tool | Role |
|------|------|
| Python 3.11 | Core language |
| Scikit-learn | TF-IDF + FeatureUnion + Logistic Regression + cross-validation |
| LIME | Per-prediction word-level explainability |
| Streamlit | Multi-page interactive web UI |
| Plotly | Confusion matrix, feature importance, language distribution charts |
| Pandas | Dataset loading, EDA, external data ingestion |
| Joblib | Model serialisation |
| re (regex) | Scam signal detection, sender validation, language detection |

---

## Project Structure

```
UPI-SCAM-SMS-DETECTOR/
├── app.py                  # Main Streamlit app (detector UI)
├── pages/
│   └── Under_the_Hood.py   # Full technical breakdown for recruiters
├── preprocess.py           # Text cleaner — preserves 8 Indian Unicode blocks
├── train.py                # Model training script
├── generate_dataset.py     # Dataset builder (synthetic + external sources)
├── data/
│   └── dataset.csv         # 2,394 labelled SMS (5 languages)
├── models/
│   └── upi_scam_detector.pkl  # Trained pipeline (TF-IDF + LR)
├── requirements.txt
└── .streamlit/
    └── config.toml         # Dark theme config
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/UPI-SCAM-SMS-DETECTOR.git
cd UPI-SCAM-SMS-DETECTOR

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate dataset — downloads external Hindi/Telugu data
python generate_dataset.py

# 4. (Optional) Retrain the model
python train.py

# 5. Launch the app
streamlit run app.py
```

> The pre-trained model and dataset are already committed — steps 3 and 4 are only needed if you want to regenerate from scratch.

---

## Dataset

| Source | Size | Languages |
|--------|------|-----------|
| Synthetic (hand-crafted UPI scam/legit) | ~353 messages | Tamil, Telugu, Hindi, English, Mixed |
| [princebari Indian SMS](https://github.com/princebari/-SMS-Spam-Classification-on-Indian-Dataset-A-Crowdsourced-Collection-of-Hindi-and-English-Messages) | ~2,000 messages | English (Indian context) |
| [shshnk158 Multilingual SMS](https://github.com/shshnk158/Multilingual-SMS-spam-detection-using-RNN) | filtered rows | Hindi / Telugu (Unicode script) |
| **Total** | **2,394** | **5 languages** |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~99%+ (on current dataset) |
| Precision (Scam) | ~99%+ |
| Recall (Scam) | ~99%+ |
| CV F1 (5-fold) | ~99%+ |

> Note: High accuracy reflects consistent vocabulary patterns in the synthetic portion. Real-world performance on unseen SMS will vary. See the **Limitations** tab inside the app for a full discussion.

---

## Key Engineering Decisions

- **No language-specific tokeniser** — Tamil, Telugu, and Hindi all use spaces between words, so standard word tokenisation works across all 5 languages without branching logic.
- **Char n-grams (2–4)** — handles agglutinative Tamil and Telugu morphology without a stemmer.
- **FeatureUnion** — word TF-IDF (15k features) + char TF-IDF (20k features) concatenated into a single sparse matrix.
- **Rule-based layers run in parallel** — the ML model sees preprocessed tokens; the signal detector and sender checker run on raw text, giving two independent evidence streams.
- **Combined verdict** — a 6-case decision matrix merges content + sender signals so the user never has to reconcile contradicting results.

---

## Scam Patterns Covered (13 types)

KYC threats · Account suspension · Fake OTP requests · Prize/lottery phishing · Fake government schemes · Fraudulent refunds · Fake bank alerts · Courier fee scams · Job/WFH scams · Loan scams · SIM expiry threats · Crypto/investment traps · Social engineering

---

## Limitations

- Synthetic training data — model has not been validated on real-world UPI SMS logs
- Static sender allowlist — new legitimate senders may be flagged as Suspicious
- No semantic understanding — novel phishing tactics with new vocabulary may slip through
- No real-time deployment — run locally or deploy to Streamlit Community Cloud

---

## Concepts Demonstrated

`Binary Classification` · `TF-IDF` · `FeatureUnion` · `Logistic Regression` · `LIME (XAI)` · `Cross-Validation` · `Multilingual NLP` · `Unicode Script Handling` · `Regex Pattern Matching` · `Rule-Based Systems` · `Streamlit Multi-Page App` · `Session State` · `Plotly Visualisation` · `Model Serialisation` · `Synthetic Data Generation` · `External Dataset Integration`

---

## Author

Built by **Megan** — fresher portfolio project showcasing NLP, ML, and product thinking.
