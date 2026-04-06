"""
UPI Scam SMS Detector — Streamlit App
Supports Tamil, Telugu, Hindi, English, and Mixed SMS messages.
"""

import os
import re

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer

from preprocess import preprocess_text

# ── Sender analyser ───────────────────────────────────────────────────────────
# Known legitimate brand/service codes used in Indian SMS sender IDs
_KNOWN_CODES = {
    # Banks
    "HDFCBK","HDFCBN","HDFCMF","HDFCSL","SBIINB","SBICRD","SBISMS","SBIUPI","SBIMF",
    "ICICIB","ICICIN","AXISBK","AXISBN","KOTAKB","INDUSB","INDUSIN",
    "YESBNK","PNBSMS","PNBNET","BOISMS","CANBNK","UNIONB","CENTBK",
    "IDBIBK","RBLBNK","FEDBKM","FEDBNK","SCBKMS","CITIBK","HSBCIN",
    "DEUBNK","DBSBNK","BAJAFN","BAJAJ","LICIND","LICIOF",
    # UPI / Wallets
    "PAYTMB","PAYTM","GPAY","PHONEPE","PHONPE","BHIMPAY","BHIM",
    "MOBIKW","FREECRG","AMAZON","AMZNIN","CREDAPP","CRED",
    # Investments / Finance
    "ZERODHA","ZEROD","GROWWAP","GROWW","NIPPON","MIRAE","ANGELB","UPSTOX",
    "POLICYB","PBAZAAR","NPSCAN","HDFCMF","SBIMF","AXISMF",
    # Telecom
    "AIRTEL","AIRTELIN","JIONET","JIOIND","BSNLSM","BSNL",
    "VODAIN","VODAFONE","IDEACEL","IDEA","TATASKY","TATADTH",
    # E-commerce / Travel / Services
    "FLIPKRT","SWIGGY","ZOMATO","UBER","OLA","IRCTC","IRTCSM",
    "MMTRIP","MAKEMY","CLEARTR","YATRA","OLAMONY","NYKAAFS","NYKAA",
    "MEESHO","MYNTRA","BIGBSK","ZEPTO","DUNZO",
    # Government
    "GOVTIN","INCOMETAX","ITDEPT","EPFOHO","EPFINDIA","UIDAI","NSDL","CDSL",
}
# TRAI DLT-registered two-letter header prefixes
_DLT_PREFIXES = {"VM","VD","AD","DM","BK","TM","IM","TA","TB","JD","JK","CP","BP","FP","HP"}


def analyze_sender(sender: str) -> dict | None:
    """Rule-based sender credibility check for Indian UPI SMS."""
    if not sender or not sender.strip():
        return None

    raw = sender.strip()
    s = raw.upper()
    reasons, flags = [], []
    score = 0

    # 1. Personal mobile number → always suspicious for bank/UPI SMS
    if re.fullmatch(r"[6-9]\d{9}", s):
        flags.append("Sent from a 10-digit personal mobile number.")
        flags.append("Banks and UPI services NEVER send SMS from mobile numbers.")
        score -= 4

    # 2. Short code (4-6 digits) — typical service short codes
    elif re.fullmatch(r"\d{4,6}", s):
        reasons.append("Numeric short code — commonly used by TRAI-registered services.")
        score += 1

    # 3. Standard DLT format  PREFIX-BRANDCODE
    elif re.match(r"^[A-Z]{2}-[A-Z0-9]+$", s):
        prefix, code = s[:2], s[3:]
        if prefix in _DLT_PREFIXES:
            reasons.append(f"Valid DLT header prefix '{prefix}-' — TRAI-approved format.")
            score += 2
        else:
            flags.append(f"Prefix '{prefix}-' is not a standard TRAI DLT prefix.")
            score -= 1
        if code in _KNOWN_CODES or any(k in code for k in _KNOWN_CODES):
            reasons.append(f"Brand code '{code}' matches a known bank / service.")
            score += 3
        else:
            flags.append(f"Brand code '{code}' is not in the known legitimate sender list.")
            score -= 1

    # 4. Plain alphanumeric (no hyphen)
    elif re.fullmatch(r"[A-Z0-9]+", s):
        if s in _KNOWN_CODES or any(k in s for k in _KNOWN_CODES):
            reasons.append("Recognised as a known legitimate bank / service sender ID.")
            score += 3
        else:
            flags.append("Alphanumeric ID does not match any known legitimate service.")
            score -= 1

    # 5. Anything else (special chars, spaces, etc.)
    else:
        flags.append("Unusual format — does not match any standard SMS sender pattern.")
        score -= 2

    if score >= 3:
        verdict, color, icon = "Legitimate", "#00C851", "✅"
    elif score >= 0:
        verdict, color, icon = "Suspicious", "#FFA500", "⚠️"
    else:
        verdict, color, icon = "Likely Fake", "#FF4B4B", "🚨"

    return {
        "verdict": verdict, "color": color, "icon": icon,
        "reasons": reasons, "flags": flags, "raw": raw,
    }

# ── Scam signal detector ──────────────────────────────────────────────────────
_URGENCY       = re.compile(r"\b(urgent|immediately|now|asap|hurry|expire[sd]?|last.?chance|today.?only)\b", re.I)
_OTP_PIN       = re.compile(
    r"\b("
    r"share.{0,20}(otp|pin|cvv|password)"   # "share your OTP/PIN"
    r"|send.{0,20}(otp|pin)"                 # "send OTP"
    r"|enter.{0,20}(otp|pin)"               # "enter OTP"
    r"|provide.{0,20}(otp|pin)"             # "provide OTP"
    r"|give.{0,20}(otp|pin)"               # "give your PIN"
    r"|confirm.{0,20}pin"                   # "confirm your PIN"
    r"|otp.{0,30}(share|send|required|verify|claim|transfer|receive|unlock|activate)"
    r"|pin.{0,20}(share|send|required|verify|unlock|activate)"
    r"|upi.{0,10}pin"                        # "UPI PIN" — always a red flag in scam context
    r"|cvv\b"                                # CVV by itself is always suspicious
    r")", re.I
)
_KYC_BLOCK     = re.compile(r"\b(kyc|blocked?|suspend|deactivat|frozen?|locked?)\b", re.I)
_PRIZE_LOTTERY = re.compile(r"\b(won|winner|prize|lottery|reward|congratul|claim|lucky)\b", re.I)
_PHONE_IN_BODY = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")
_URL_IN_BODY   = re.compile(r"(https?://|www\.|bit\.ly/|tinyurl\.com)\S+", re.I)
_CALL_ACTION   = re.compile(r"\b(call|contact|click|tap|visit|open|download|install)\b.{0,30}(link|here|now|below|number|app)\b", re.I)
_REFUND_TRICK  = re.compile(r"\b(refund|cashback|electricity.?bill|gas.?bill)\b.{0,40}\b(upi|pay|send|transfer)\b", re.I)


def detect_language(text: str) -> tuple[str, str]:
    """Detect SMS language from Unicode script. Returns (label, flag emoji)."""
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    telugu     = len(re.findall(r"[\u0C00-\u0C7F]", text))
    tamil      = len(re.findall(r"[\u0B80-\u0BFF]", text))
    latin      = len(re.findall(r"[a-zA-Z]", text))

    if devanagari > 5 and latin > 5:
        return "Mixed (Hindi + English)", "🌐"
    if devanagari > 5:
        return "Hindi", "🇮🇳"
    if telugu > 5 and latin > 5:
        return "Mixed (Telugu + English)", "🌐"
    if telugu > 5:
        return "Telugu", "🇮🇳"
    if tamil > 5 and latin > 5:
        return "Mixed (Tamil + English)", "🌐"
    if tamil > 5:
        return "Tamil", "🇮🇳"
    return "English", "🇬🇧"


def detect_scam_signals(text: str) -> list[dict]:
    """
    Scan raw SMS text for explicit scam red-flag patterns.
    Returns a list of dicts: {label, detail, severity}  severity ∈ high/medium
    """
    signals = []

    if m := _URGENCY.search(text):
        signals.append({
            "label": "Urgency / Pressure language",
            "detail": f"Found: '{m.group()}' — scammers create panic to stop you thinking clearly.",
            "severity": "high",
        })
    if m := _OTP_PIN.search(text):
        signals.append({
            "label": "OTP / PIN request",
            "detail": f"Found: '{m.group()}' — no legitimate bank or service will ever ask for your OTP or PIN.",
            "severity": "high",
        })
    if m := _KYC_BLOCK.search(text):
        signals.append({
            "label": "Account block / KYC threat",
            "detail": f"Found: '{m.group()}' — classic social-engineering threat to trigger panic.",
            "severity": "high",
        })
    if m := _PRIZE_LOTTERY.search(text):
        signals.append({
            "label": "Fake prize / lottery",
            "detail": f"Found: '{m.group()}' — unsolicited prize messages are almost always phishing.",
            "severity": "high",
        })
    phones = _PHONE_IN_BODY.findall(text)
    if phones:
        signals.append({
            "label": "Phone number embedded in message",
            "detail": (
                f"Number(s) found: {', '.join(phones)}. "
                "If this is your own number (e.g. in a recharge confirmation) it is normal. "
                "If it is an unknown number asking you to call back, do NOT call — scammers staff these lines."
            ),
            "severity": "medium",
        })
    urls = _URL_IN_BODY.findall(text)
    if urls:
        signals.append({
            "label": "URL / link in message",
            "detail": f"Link detected — never click links in unsolicited SMS. Always visit your bank's app directly.",
            "severity": "high",
        })
    if m := _CALL_ACTION.search(text):
        signals.append({
            "label": "Call-to-action directive",
            "detail": f"Found: '{m.group()}' — pressures you to take immediate action.",
            "severity": "medium",
        })
    if m := _REFUND_TRICK.search(text):
        signals.append({
            "label": "Refund-via-UPI trick",
            "detail": f"Found: '{m.group()}' — scammers pose as utility companies offering refunds to harvest UPI credentials.",
            "severity": "high",
        })

    return signals


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Scam SMS Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .scam-box  { background:#FF4B4B22; border:2px solid #FF4B4B;
                 border-radius:10px; padding:16px; text-align:center; }
    .legit-box { background:#00C85122; border:2px solid #00C851;
                 border-radius:10px; padding:16px; text-align:center; }
    .big-label { font-size:2rem; font-weight:700; }
    .prob-row  { display:flex; gap:12px; margin-top:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "models/upi_scam_detector.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def get_explainer():
    return LimeTextExplainer(class_names=["Legitimate", "Scam"])


# ── Example messages ──────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "tag": "🚨 Scam — English",
        "sender": "9876543210",
        "text": (
            "URGENT: Your UPI account will be BLOCKED! Update KYC immediately. "
            "Call 9876543210 now!"
        ),
    },
    {
        "tag": "✅ Legit — English",
        "sender": "VM-HDFCBK",
        "text": (
            "Rs 5000.00 credited to your account XXXXXXXX1234 via UPI on "
            "03-Apr-2026. Ref: UPI123456789"
        ),
    },
    {
        "tag": "🚨 Scam — Tamil",
        "sender": "9123456789",
        "text": (
            "அவசரம்: உங்கள் UPI கணக்கு தடுக்கப்படும். KYC புதுப்பிக்க "
            "இப்போதே அழைக்கவும் 9876543210"
        ),
    },
    {
        "tag": "✅ Legit — Tamil",
        "sender": "VM-SBIINB",
        "text": (
            "Rs 5000.00 உங்கள் கணக்கில் XXXXXXXX1234 UPI மூலம் வரவு "
            "வைக்கப்பட்டது. Ref: UPI123456"
        ),
    },
    {
        "tag": "🚨 Scam — Hindi",
        "sender": "8899001122",
        "text": (
            "तुरंत: आपका UPI खाता ब्लॉक हो जाएगा! KYC अपडेट करें। "
            "अभी कॉल करें 9876543210"
        ),
    },
    {
        "tag": "✅ Legit — Hindi",
        "sender": "VM-SBIINB",
        "text": (
            "Rs 5000.00 आपके खाते XXXXXXXX1234 में UPI द्वारा जमा किया गया। "
            "Ref: UPI123456"
        ),
    },
    {
        "tag": "🚨 Scam — Telugu",
        "sender": "9000112233",
        "text": (
            "అత్యవసరం: మీ UPI ఖాతా బ్లాక్ అవుతుంది! KYC అప్‌డేట్ చేయండి. "
            "ఇప్పుడే కాల్ చేయండి 9876543210"
        ),
    },
    {
        "tag": "✅ Legit — Telugu",
        "sender": "VM-HDFCBK",
        "text": (
            "Rs 5000.00 మీ ఖాతా XXXXXXXX1234 లో UPI ద్వారా జమ చేయబడింది. "
            "Ref: UPI123456"
        ),
    },
    {
        "tag": "🚨 Scam — Mixed",
        "sender": "8765432109",
        "text": (
            "URGENT: உங்கள் account block ஆகும். Click here to update KYC now! "
            "Call 9876543210 immediately"
        ),
    },
    {
        "tag": "✅ Legit — Mixed",
        "sender": "AD-AIRTEL",
        "text": (
            "Payment successful! Rs 500 உங்கள் கணக்கில் credited ஆகியது. "
            "Ref: TXN123456789"
        ),
    },
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center; padding: 24px 0 8px 0;">
        <svg width="170" height="210" viewBox="0 0 170 210" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#6655ff"/>
              <stop offset="100%" stop-color="#bb44ff"/>
            </linearGradient>
            <clipPath id="shieldClip">
              <path d="M85 12 L152 42 L152 100 Q152 158 85 183 Q18 158 18 100 L18 42 Z"/>
            </clipPath>
          </defs>

          <!-- Outer pulse ring -->
          <path d="M85 12 L152 42 L152 100 Q152 158 85 183 Q18 158 18 100 L18 42 Z"
                fill="#7c55ff" opacity="0.08">
            <animate attributeName="opacity" values="0.04;0.18;0.04" dur="2.2s" repeatCount="indefinite"/>
          </path>

          <!-- Shield body -->
          <path d="M85 12 L152 42 L152 100 Q152 158 85 183 Q18 158 18 100 L18 42 Z"
                fill="url(#shieldGrad)" opacity="0.88"/>

          <!-- Shield inner highlight -->
          <path d="M85 22 L142 48 L142 100 Q142 150 85 172 Q28 150 28 100 L28 48 Z"
                fill="none" stroke="white" stroke-width="1" opacity="0.15"/>

          <!-- Shield border pulse -->
          <path d="M85 12 L152 42 L152 100 Q152 158 85 183 Q18 158 18 100 L18 42 Z"
                fill="none" stroke="#aaaaff" stroke-width="2">
            <animate attributeName="opacity" values="0.3;0.9;0.3" dur="2.2s" repeatCount="indefinite"/>
          </path>

          <!-- Scan line + trail (clipped to shield) -->
          <g clip-path="url(#shieldClip)">
            <!-- glow trail below scan -->
            <rect x="18" y="42" width="134" height="22" fill="#00ffcc" opacity="0.07">
              <animate attributeName="y" values="42;168;42" dur="3s" repeatCount="indefinite" calcMode="ease-in-out"/>
            </rect>
            <!-- scan line -->
            <line x1="18" y1="42" x2="152" y2="42" stroke="#00ffcc" stroke-width="2.5" stroke-linecap="round" opacity="0.95">
              <animate attributeName="y1" values="42;170;42" dur="3s" repeatCount="indefinite" calcMode="ease-in-out"/>
              <animate attributeName="y2" values="42;170;42" dur="3s" repeatCount="indefinite" calcMode="ease-in-out"/>
            </line>
          </g>

          <!-- Lock shackle -->
          <path d="M69 94 Q69 76 85 76 Q101 76 101 94"
                fill="none" stroke="white" stroke-width="5" stroke-linecap="round" opacity="0.95"/>

          <!-- Lock body -->
          <rect x="63" y="92" width="44" height="32" rx="6" fill="white" opacity="0.93"/>

          <!-- Keyhole -->
          <circle cx="85" cy="108" r="5" fill="#6655ff"/>
          <rect x="83" y="111" width="4" height="7" rx="2" fill="#6655ff"/>

          <!-- SCANNING label -->
          <text x="85" y="200" text-anchor="middle"
                font-family="monospace" font-size="10" fill="#9988cc" letter-spacing="3">
            <animate attributeName="opacity" values="1;0.15;1" dur="1.6s" repeatCount="indefinite"/>
            ● SCANNING
          </text>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🛡️ UPI Scam Detector")
    st.caption("Tamil · Telugu · Hindi · English · Mixed")
    st.divider()

    st.markdown("**What this app checks:**")
    checks = [
        ("🤖", "ML model", "Logistic Regression on TF-IDF features"),
        ("🚩", "8 Red-flag patterns", "Urgency, OTP, KYC, prize, URLs…"),
        ("📨", "Sender ID", "60+ known bank & service codes"),
        ("⚖️", "Combined verdict", "All signals merged into one conclusion"),
    ]
    for icon, title, desc in checks:
        st.markdown(
            f"""<div style="padding:7px 0;border-bottom:1px solid #2e3250;">
            <span style="font-size:1.1rem;">{icon}</span>
            <strong style="color:#e0e0e0;"> {title}</strong>
            <div style="color:#888;font-size:0.78rem;margin-left:1.6rem;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Languages supported:**")
    st.markdown("🇮🇳 Tamil · Telugu · Hindi")
    st.markdown("🇬🇧 English &nbsp;|&nbsp; 🌐 Mixed")
    st.divider()
    st.markdown("**Dataset:**")
    st.markdown("2,394 samples · synthetic + real-world · 5 languages")
    st.divider()
    st.caption("Built with Scikit-learn · LIME · Streamlit")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🛡️ UPI Scam SMS Detector")
st.markdown(
    "Paste any UPI-related SMS below — the model will flag it as **Scam** or **Legitimate** "
    "and explain *why* using LIME."
)
st.divider()

# Load model
model = load_model()
explainer = get_explainer()

if model is None:
    st.error(
        "**Model not found.**  \n"
        "Run `python generate_dataset.py` then `python train.py` to create it."
    )
    st.stop()

# Session state
if "sms_input" not in st.session_state:
    st.session_state["sms_input"] = ""
if "sender_input" not in st.session_state:
    st.session_state["sender_input"] = ""

# ── Example buttons ───────────────────────────────────────────────────────────
# Each column = one language; Scam button on top, Legit button below
st.markdown("**Try an example:**")
_PAIRS = [
    ("English", EXAMPLES[0], EXAMPLES[1]),
    ("Hindi",   EXAMPLES[4], EXAMPLES[5]),
    ("Tamil",   EXAMPLES[2], EXAMPLES[3]),
    ("Telugu",  EXAMPLES[6], EXAMPLES[7]),
    ("Mixed",   EXAMPLES[8], EXAMPLES[9]),
]
_ex_cols = st.columns(5)
for col, (lang, scam_ex, legit_ex) in zip(_ex_cols, _PAIRS):
    with col:
        st.caption(lang)
        if st.button(scam_ex["tag"], use_container_width=True, key=f"ex_scam_{lang}"):
            st.session_state["sms_input"] = scam_ex["text"]
            st.session_state["sender_input"] = scam_ex.get("sender", "")
            st.rerun()
        if st.button(legit_ex["tag"], use_container_width=True, key=f"ex_legit_{lang}"):
            st.session_state["sms_input"] = legit_ex["text"]
            st.session_state["sender_input"] = legit_ex.get("sender", "")
            st.rerun()

st.markdown("")

# ── Sender + SMS inputs ───────────────────────────────────────────────────────
sender_text: str = st.text_input(
    "📨 Sender Name / Number (who sent the SMS):",
    value=st.session_state["sender_input"],
    placeholder="e.g.  VM-HDFCBK  or  AD-AIRTEL  or  9876543210",
    key="sender_input_field",
)

sms_text: str = st.text_area(
    "💬 Paste or type the SMS message:",
    value=st.session_state["sms_input"],
    height=130,
    placeholder="Enter SMS in Tamil, Hindi, Telugu, English, or mixed…",
    key="sms_textarea",
)

analyze = st.button("🔍 Analyze SMS", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze:
    if not sms_text.strip():
        st.warning("Please enter an SMS message first.")
    else:
        with st.spinner("Analyzing…"):
            processed = preprocess_text(sms_text)
            proba = model.predict_proba([processed])[0]
            pred = int(np.argmax(proba))
            scam_prob = float(proba[1])
            legit_prob = float(proba[0])
            sender_result = analyze_sender(sender_text)
            scam_signals = detect_scam_signals(sms_text)
            lang_label, lang_flag = detect_language(sms_text)

        st.divider()

        # ── Language detection badge ──────────────────────────────────────────
        st.markdown(
            f"""<div style="display:inline-block;background:#6655ff22;border:1px solid #6655ff;
                            border-radius:20px;padding:5px 16px;font-size:0.88rem;margin-bottom:12px;">
              {lang_flag} <strong>Detected language:</strong> {lang_label}
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Two signal cards side-by-side ─────────────────────────────────────
        sig_col1, sig_col2 = st.columns(2)

        # Signal 1 — Sender
        with sig_col1:
            if sender_result:
                s = sender_result
                st.markdown(
                    f"""
                    <div style="background:{s['color']}22;border:2px solid {s['color']};
                                border-radius:10px;padding:12px 14px;height:100%;">
                      <div style="font-weight:700;font-size:0.95rem;margin-bottom:6px;">
                        📨 Sender Check
                      </div>
                      <div style="font-size:1.05rem;font-weight:700;color:{s['color']};">
                        {s['icon']} {s['verdict']}
                      </div>
                      <div style="font-size:0.82rem;margin-top:4px;color:#ccc;">
                        <code>{s['raw']}</code>
                      </div>
                      <div style="margin-top:8px;font-size:0.82rem;">
                        {"".join(f'<div style="color:#00c853;">✔ {r}</div>' for r in s["reasons"])}
                        {"".join(f'<div style="color:#ff6b6b;">✘ {f}</div>' for f in s["flags"])}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background:#33333322;border:2px solid #555;
                                border-radius:10px;padding:12px 14px;">
                      <div style="font-weight:700;font-size:0.95rem;">📨 Sender Check</div>
                      <div style="color:#888;font-size:0.88rem;margin-top:6px;">No sender entered</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Signal 2 — SMS Content
        with sig_col2:
            content_color = "#FF4B4B" if pred == 1 else "#00C851"
            content_verdict = "Scam Content" if pred == 1 else "Legit Content"
            content_icon = "⚠️" if pred == 1 else "✅"
            content_detail = (
                f"Model confidence: {scam_prob:.0%} scam"
                if pred == 1
                else f"Model confidence: {legit_prob:.0%} legit"
            )
            st.markdown(
                f"""
                <div style="background:{content_color}22;border:2px solid {content_color};
                            border-radius:10px;padding:12px 14px;height:100%;">
                  <div style="font-weight:700;font-size:0.95rem;margin-bottom:6px;">
                    💬 Content Check
                  </div>
                  <div style="font-size:1.05rem;font-weight:700;color:{content_color};">
                    {content_icon} {content_verdict}
                  </div>
                  <div style="font-size:0.82rem;color:#ccc;margin-top:4px;">
                    {content_detail}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Combined final verdict ────────────────────────────────────────────
        sender_v = sender_result["verdict"] if sender_result else "Unknown"

        # Decision matrix
        if pred == 1 and sender_v == "Likely Fake":
            final_verdict  = "SCAM"
            final_color    = "#FF4B4B"
            final_icon     = "🚨"
            final_headline = "Fake sender + scam content — do not engage."
            final_reasons  = [
                "The sender is not a registered bank or service — it is a personal or unknown number.",
                "The SMS content itself contains scam patterns.",
                "Real banks (SBI, HDFC, etc.) always use registered DLT sender IDs like VM-HDFCBK.",
            ]
        elif pred == 1 and sender_v == "Suspicious":
            final_verdict  = "SCAM"
            final_color    = "#FF4B4B"
            final_icon     = "🚨"
            final_headline = "Unverified sender with scam content — treat as fraud."
            final_reasons  = [
                "The sender ID does not match any known bank or UPI service.",
                "The message content shows scam signals (urgency, threats, requests for action).",
            ]
        elif pred == 1 and sender_v == "Legitimate":
            final_verdict  = "SCAM"
            final_color    = "#FF4B4B"
            final_icon     = "🚨"
            final_headline = "Sender looks real but content is scam — possible sender spoofing."
            final_reasons  = [
                "Scammers can fake or spoof legitimate sender IDs (e.g. VM-HDFCBK).",
                "The message content contains scam patterns regardless of the sender name.",
                "Never click links or call numbers from such messages — verify directly with your bank.",
            ]
        elif pred == 0 and sender_v == "Likely Fake":
            final_verdict  = "SUSPICIOUS"
            final_color    = "#FFA500"
            final_icon     = "⚠️"
            final_headline = "Content looks normal but the sender is fake — possible social engineering."
            final_reasons  = [
                "Scammers sometimes send normal-looking messages first to build trust before striking.",
                "The sender is a personal mobile number — legitimate banks never do this.",
                "Do not reply, click links, or share OTPs even if the message looks harmless.",
            ]
        elif pred == 0 and sender_v == "Suspicious":
            final_verdict  = "SUSPICIOUS"
            final_color    = "#FFA500"
            final_icon     = "⚠️"
            final_headline = "Content seems fine but sender is unverified — proceed with caution."
            final_reasons  = [
                "The sender ID is not recognised as a registered bank or UPI service.",
                "Verify the transaction/information directly through your bank's official app or website.",
            ]
        else:
            # Legit content + Legit/Unknown sender
            final_verdict  = "LEGITIMATE"
            final_color    = "#00C851"
            final_icon     = "✅"
            final_headline = "Both sender and content appear genuine."
            final_reasons  = [
                "Sender matches a known TRAI-registered bank or service ID." if sender_v == "Legitimate"
                else "Message content does not show any scam patterns.",
                "No urgency, threats, or suspicious links detected.",
            ]

        st.markdown(
            f"""
            <div style="background:{final_color}18;border:2.5px solid {final_color};
                        border-radius:12px;padding:18px 20px;margin-top:4px;">
              <div style="font-size:1.5rem;font-weight:800;color:{final_color};margin-bottom:6px;">
                {final_icon} FINAL VERDICT: {final_verdict}
              </div>
              <div style="font-size:0.97rem;font-weight:600;margin-bottom:10px;color:#e0e0e0;">
                {final_headline}
              </div>
              {"".join(f'<div style="font-size:0.87rem;color:#ccc;margin-bottom:3px;">• {r}</div>' for r in final_reasons)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ── Probability metrics ───────────────────────────────────────────────
        col1, col2 = st.columns(2)
        col1.metric("Scam Probability", f"{scam_prob:.1%}")
        col2.metric("Legitimate Probability", f"{legit_prob:.1%}")

        risk_label = (
            "🔴 High Risk" if scam_prob > 0.70
            else "🟠 Medium Risk" if scam_prob > 0.40
            else "🟢 Low Risk"
        )
        st.progress(scam_prob, text=f"Scam Risk  —  {risk_label}")

        # ── Scam signal checklist ─────────────────────────────────────────────
        st.divider()
        st.markdown("### 🚩 Red Flag Checklist")
        if scam_signals:
            for sig in scam_signals:
                color  = "#ff4b4b" if sig["severity"] == "high" else "#FFA500"
                badge  = "HIGH" if sig["severity"] == "high" else "MEDIUM"
                st.markdown(
                    f"""
                    <div style="background:{color}18;border-left:4px solid {color};
                                border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                      <span style="background:{color};color:#fff;font-size:0.72rem;
                                   font-weight:700;padding:2px 7px;border-radius:3px;
                                   margin-right:8px;">{badge}</span>
                      <strong style="color:#e0e0e0;">{sig['label']}</strong>
                      <div style="color:#bbb;font-size:0.85rem;margin-top:4px;">{sig['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No specific red-flag patterns detected in the message text.")

        # ── Safety tips (only for scam / suspicious) ──────────────────────────
        if final_verdict in ("SCAM", "SUSPICIOUS"):
            st.divider()
            st.markdown("### 🛡️ What To Do Next")
            tips = [
                ("Do NOT call any number mentioned in the SMS",
                 "Scammers staff these lines. Call your bank only on the number on the back of your card or the official website."),
                ("Do NOT click any link",
                 "Even if the URL looks official — scammers use lookalike domains. Always open your bank app directly."),
                ("Do NOT share OTP, PIN, or CVV",
                 "No legitimate bank, NPCI, or government body will ever ask for these over SMS or phone."),
                ("Report it",
                 "Forward the SMS to 1930 (National Cyber Crime helpline) or report at cybercrime.gov.in."),
                ("Block the sender",
                 "Block the number on your phone and report it as spam to your telecom provider."),
            ]
            for title, body in tips:
                st.markdown(
                    f"""
                    <div style="background:#1a1d27;border-radius:8px;
                                padding:10px 14px;margin-bottom:7px;
                                border:1px solid #2e3250;">
                      <strong style="color:#e0e0e0;">🔒 {title}</strong>
                      <div style="color:#aaa;font-size:0.86rem;margin-top:3px;">{body}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── LIME explanation ──────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔬 Explanation (LIME)")
        st.caption(
            "Words highlighted in **orange/red** push toward Scam; "
            "**blue/green** push toward Legitimate."
        )

        with st.spinner("Generating LIME explanation…"):
            try:
                exp = explainer.explain_instance(
                    processed,
                    model.predict_proba,
                    num_features=12,
                    num_samples=500,
                )
                html = exp.as_html()
                # Restyle LIME HTML to match the dark app theme
                fix_css = """
<style>
  html, body {
    background-color: #0e1117 !important;
    color: #e0e0e0 !important;
    font-family: sans-serif !important;
    margin: 0; padding: 8px;
  }
  div, p, li, ul, h1, h2, h3, h4, th, td {
    color: #e0e0e0 !important;
    background-color: transparent !important;
  }
  table {
    background-color: #1a1d27 !important;
    border-radius: 8px;
    overflow: hidden;
  }
  td, th { border-color: #2e3250 !important; padding: 6px 10px !important; }
  .lime { background-color: #1a1d27 !important; border-radius: 10px; padding: 10px; }
  p[style] {
    background-color: #1a1d27 !important;
    border-radius: 8px;
    padding: 12px !important;
    line-height: 2.2 !important;
    font-size: 15px !important;
  }
  /* Override LIME's default orange/blue with vivid red/green */
  span[style*="background: rgba(255, 165"] ,
  span[style*="background: rgb(255, 165"]  { background-color: #ff3b3b !important; color: #fff !important; font-weight: 700 !important; border-radius: 4px; padding: 2px 5px; }
  span[style*="background: rgba(0, 128"]  ,
  span[style*="background: rgb(0, 128"]   { background-color: #00c853 !important; color: #fff !important; font-weight: 700 !important; border-radius: 4px; padding: 2px 5px; }
  /* Catch all coloured spans and ensure text is readable */
  span[style*="background"] { color: #fff !important; border-radius: 4px; padding: 2px 5px; font-weight: 600; }
  svg text { fill: #cccccc !important; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0e1117; }
  ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
</style>
"""
                # Inject a colour legend right after <body>
                legend_html = """
<div style="display:flex;gap:20px;align-items:center;background:#1a1d27;
            border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:13px;">
  <strong style="color:#e0e0e0;">Word colour key:</strong>
  <span style="background:#ff3b3b;color:#fff;border-radius:4px;
               padding:3px 10px;font-weight:700;">RED = pushes toward SCAM</span>
  <span style="background:#00c853;color:#fff;border-radius:4px;
               padding:3px 10px;font-weight:700;">GREEN = pushes toward LEGITIMATE</span>
  <span style="color:#aaa;">Darker shade = stronger influence</span>
</div>
"""
                html = html.replace("</head>", fix_css + "</head>", 1)
                html = html.replace("<body>", "<body>" + legend_html, 1)
                components.html(html, height=480, scrolling=True)

                # Top features as a readable list
                feature_weights = exp.as_list(label=1)
                if feature_weights:
                    st.markdown("**Top contributing words / patterns:**")
                    for feat, weight in feature_weights[:10]:
                        icon = "🔴" if weight > 0 else "🟢"
                        direction = f"+{weight:.3f}" if weight > 0 else f"{weight:.3f}"
                        st.markdown(f"- `{feat}` &nbsp; {icon} `{direction}`")
            except Exception as exc:
                st.warning(f"LIME explanation could not be generated: {exc}")

        # ── Technical details ─────────────────────────────────────────────────
        with st.expander("🔧 Technical details — preprocessed text"):
            st.caption(
                "This is the cleaned text the ML model actually sees after preprocessing "
                "(lowercased, URLs tokenised, punctuation stripped)."
            )
            st.code(processed, language=None)
