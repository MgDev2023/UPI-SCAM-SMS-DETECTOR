# UPI Scam SMS Detector

A web app that checks if a UPI-related SMS is a scam or not. It supports 5 languages — Tamil, Telugu, Hindi, English, and Mixed.

**Live App:** [Click here to try it](https://upi-scam-sms-detector-ghgwrj6c5xdjatsqqmqsze.streamlit.app/)

---

## What does it do?

You paste an SMS message into the app and it tells you:
- Is this a **scam** or **legit**?
- **Why** it thinks so (which words made it suspicious)
- Whether the **sender ID** looks real or fake
- **Safety tips** if it's a scam

---

## How it works

I built 3 layers of checks:

1. **ML Model** — trained on 2,394 SMS messages to classify scam vs legit
2. **Red Flag Detector** — looks for common scam tricks like OTP requests, fake prizes, KYC threats, etc.
3. **Sender Verifier** — checks if the sender ID is a known bank/service or looks suspicious

All 3 results are combined to give one final verdict.

---

## Tech used

- Python
- Scikit-learn (machine learning)
- Streamlit (web app)
- LIME (explains which words triggered the model)
- Regex (pattern matching for scam signals)

---

## How to run it locally

```bash
git clone https://github.com/MgDev2023/UPI-SCAM-SMS-DETECTOR.git
cd UPI-SCAM-SMS-DETECTOR
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

- 2,394 labeled SMS messages
- Covers 5 languages: Tamil, Telugu, Hindi, English, Mixed
- Mix of hand-crafted scam examples and publicly available datasets

---

## Model accuracy

~99% on test data. (Note: this is on the training dataset — real-world results may vary since the training data is mostly synthetic.)

---

## Scam types it can detect

KYC threats, fake OTPs, prize scams, fake bank alerts, job scams, loan scams, courier fee scams, account suspension threats, and more.

---

## Made by

Megan — this is one of my fresher portfolio projects to practice ML and NLP.
