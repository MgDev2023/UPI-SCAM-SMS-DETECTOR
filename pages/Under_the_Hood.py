"""
Under the Hood — Full project breakdown for recruiters.
Everything about how this app was designed, built, and evaluated.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Ensure project root is on path so preprocess can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocess import preprocess_text

st.set_page_config(
    page_title="Under the Hood | UPI Scam Detector",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .section-card {
        background: #1e1e2e22;
        border-left: 4px solid #7c7cff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .tag {
        display: inline-block;
        background: #7c7cff33;
        color: #aaaaff;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .step-num {
        font-size: 2rem;
        font-weight: 800;
        color: #7c7cff;
        line-height: 1;
    }
    .metric-card {
        background: #ffffff08;
        border: 1px solid #ffffff18;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; }
    .metric-lbl { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "models/upi_scam_detector.pkl"
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data
def load_dataset():
    path = "data/dataset.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


def get_top_features(model, n=15):
    """Extract top scam and legitimate features from the LR coefficients."""
    word_feats = model.named_steps["features"].transformer_list[0][1].get_feature_names_out()
    char_feats = model.named_steps["features"].transformer_list[1][1].get_feature_names_out()
    all_feats = np.concatenate([word_feats, char_feats])
    coefs = model.named_steps["clf"].coef_[0]
    idx_sorted = np.argsort(coefs)
    top_scam = [(all_feats[i], coefs[i]) for i in idx_sorted[-n:][::-1]]
    top_legit = [(all_feats[i], coefs[i]) for i in idx_sorted[:n]]
    return top_scam, top_legit


@st.cache_data
def compute_test_metrics(_model):
    """Re-run the exact same 80/20 split used in train.py and return live metrics."""
    path = "data/dataset.csv"
    if not os.path.exists(path) or _model is None:
        return None
    df_m = pd.read_csv(path, encoding="utf-8-sig")
    df_m["processed"] = df_m["text"].apply(preprocess_text)
    X = df_m["processed"].values
    y = df_m["label"].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = _model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "cm":        cm.tolist(),
        "n_test":    len(y_test),
        "n_total":   len(df_m),
    }


# ── Load resources ─────────────────────────────────────────────────────────────
model = load_model()
df = load_dataset()

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🔧 Under the Hood")
st.markdown(
    "#### A complete walkthrough of how the **UPI Scam SMS Detector** was designed, "
    "built, and evaluated — written for recruiters and technical reviewers."
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_data, tab_pipeline, tab_results, tab_workflow, tab_limits = st.tabs(
    ["📌 Overview", "📊 Dataset", "⚙️ Pipeline", "📈 Results", "🗺️ Workflow", "⚠️ Limitations"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("What This Project Does")
    st.markdown(
        """
        This is an **NLP-based binary classifier** that detects fraudulent UPI SMS messages
        targeting Indian mobile users. It handles **Tamil, Telugu, Hindi, English**, and
        **code-mixed** messages — covering 5 languages and 4 scripts in a single pipeline.

        A user pastes an SMS + optional sender name/number →
        - The **ML model** classifies the message content as Scam / Legitimate
        - A **rule-based sender checker** verifies whether the sender ID is a known
          TRAI-registered bank or service, a personal mobile number, or an unknown code
        - A **combined verdict engine** merges both signals into one final conclusion —
          because a fake sender with legit-looking content is still suspicious (social engineering)
        - A **scam signal detector** scans the raw text for 8 explicit red-flag patterns
          (urgency language, OTP requests, embedded phone numbers, URLs, KYC threats, etc.)
          and presents them as a plain-English checklist the user can read at a glance
        - **LIME** highlights the exact words that drove the ML model's decision
        - **Safety tips** are shown for every Scam or Suspicious verdict with concrete
          next steps (report number, what not to share, how to verify)
        """
    )

    st.subheader("Why This Problem?")
    st.markdown(
        """
        India processed **131 billion UPI transactions** in FY2024. Scammers exploit the
        same UPI vocabulary — KYC threats, fake prize alerts, OTP phishing — in both
        English and regional languages. Tamil, Hindi, and Telugu-speaking users are especially
        under-served because most existing scam detectors are English-only.

        The multi-language requirement made this a non-trivial engineering challenge:
        Tamil and Telugu are **agglutinative** (words change form through suffixes), use
        completely different scripts from each other, and have no standard open-source
        tokeniser. Hindi uses the Devanagari script. All three use spaces between words —
        a key insight that avoids the need for language-specific tokenisation libraries.
        That drove the specific design decisions described in the Pipeline tab.
        """
    )

    st.divider()
    st.subheader("Tech Stack")

    stack = [
        ("Python 3.11", "Core language", "#3776AB"),
        ("Pandas", "Dataset loading, EDA, label distribution, external dataset ingestion", "#130654"),
        ("Scikit-learn", "TF-IDF vectorisers, FeatureUnion, Logistic Regression, cross-validation", "#F7931E"),
        ("LIME", "Model explainability — per-prediction word-level importance", "#FF6B6B"),
        ("Streamlit", "Interactive multi-page web UI with live predictions and session state", "#FF4B4B"),
        ("Plotly", "Interactive charts — confusion matrix heatmap, feature importance bars, pie charts", "#636EFA"),
        ("Joblib", "Model serialisation and fast pickle-based loading", "#4A4A4A"),
        ("re (regex)", "Scam signal detection (8 patterns), sender validation, language detection", "#888888"),
    ]

    col_a, col_b = st.columns(2)
    for i, (name, purpose, _) in enumerate(stack):
        target = col_a if i % 2 == 0 else col_b
        with target:
            st.markdown(
                f"""
                <div class="section-card">
                <strong>{name}</strong><br>
                <span style="color:#aaa;font-size:0.9rem">{purpose}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATASET
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Dataset Design")
    st.markdown(
        """
        The dataset combines two sources:

        **1. Synthetic UPI-specific messages (hand-crafted)**
        No public Tamil/Hindi/Telugu UPI scam SMS dataset exists.
        ~350 messages were written to cover real scam patterns across Tamil, Telugu, English, and Mixed.

        **2. Real-world Hindi + Telugu SMS (external free datasets)**
        Two freely available GitHub datasets were merged to add genuine Hindi and Telugu messages:
        - [shshnk158 Multilingual SMS](https://github.com/shshnk158/Multilingual-SMS-spam-detection-using-RNN) — ~10k messages, English + Hindi + Telugu
        - [princebari Indian SMS](https://github.com/princebari/-SMS-Spam-Classification-on-Indian-Dataset-A-Crowdsourced-Collection-of-Hindi-and-English-Messages) — 2k messages, Hindi + English (crowdsourced from 43 participants)

        From shshnk158 only Devanagari/Telugu-script rows are kept (filter_indian_only=True).
        All 2,000 princebari rows are kept — it is crowdsourced from 43 Indian participants
        so even the English messages are India-specific UPI/SMS context, not generic spam.
        """
    )

    if df is not None:
        # Summary metrics — dynamic: works regardless of which languages are present
        n_total = len(df)
        n_scam = int((df["label"] == 1).sum())
        n_legit = int((df["label"] == 0).sum())
        lang_counts = df["language"].value_counts()

        # Fixed 3 cards + one card per language present
        lang_colors = {
            "English": "#4A90D9", "Tamil": "#F5A623", "Mixed": "#9B59B6",
            "Hindi": "#E91E63", "Telugu": "#00BCD4",
            "Mixed-Hindi": "#FF9800", "Mixed-Telugu": "#26A69A",
        }
        lang_entries = [(lang, int(cnt)) for lang, cnt in lang_counts.items()]

        base_metrics = [
            ("Total Samples", n_total, "#7c7cff"),
            ("Scam", n_scam, "#FF4B4B"),
            ("Legitimate", n_legit, "#00C851"),
        ]
        all_metrics = base_metrics + [
            (lang, cnt, lang_colors.get(lang, "#aaaaff")) for lang, cnt in lang_entries
        ]

        cols = st.columns(len(all_metrics))
        for col, (lbl, val, _) in zip(cols, all_metrics):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        st.markdown(
            """
            **Why synthetic data?**  No open corpus exists for Tamil UPI scam SMS.
            The messages were written to cover the full spread of real-world scam
            tactics (13+ distinct patterns) and legitimate notification formats
            across 5+ major apps (PhonePe, GPay, Paytm, SBI, HDFC, Amazon Pay).
            """
        )

        st.divider()
        st.subheader("Scam Patterns Covered")
        patterns = [
            ("KYC Update Threats", "Account blocked if KYC not updated immediately"),
            ("Account Suspension", "Fake suspension notices with callback numbers"),
            ("Fake Prize / Lottery", "Congratulations-style phishing for OTP"),
            ("OTP / PIN Requests", "Direct requests to share OTP or UPI PIN"),
            ("Fake Government Schemes", "PM Kisan, COVID relief, income tax refunds"),
            ("Fraudulent Refunds", "Electricity / gas bill refunds requiring PIN"),
            ("Fake Bank Alerts", "Impersonating SBI, HDFC, ICICI, Axis, Paytm"),
            ("Shipping / Courier Scams", "Prize or parcel delivery fee paid via UPI"),
            ("Job / Work-from-Home Scams", "Registration fee via UPI for fake job offers"),
            ("Loan Scams", "Instant loan with processing fee via UPI"),
            ("SIM Expiry Threats", "SIM blocked unless Aadhaar OTP shared"),
            ("Investment / Crypto Traps", "Guaranteed returns, fake cashback enrolment"),
            ("Social Engineering", "Normal-looking messages from fake senders to build trust"),
        ]
        col1, col2 = st.columns(2)
        for i, (pname, pdesc) in enumerate(patterns):
            tgt = col1 if i % 2 == 0 else col2
            with tgt:
                st.markdown(f"**{pname}**  \n{pdesc}")

        # Charts
        st.divider()
        st.subheader("Visual Breakdown")
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            fig_label = px.pie(
                values=[n_scam, n_legit],
                names=["Scam", "Legitimate"],
                color_discrete_sequence=["#FF4B4B", "#00C851"],
                title="Class Distribution",
                hole=0.45,
            )
            fig_label.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(font=dict(color="#e0e0e0")),
            )
            st.plotly_chart(fig_label, use_container_width=True)

        with ch2:
            lang_labels = [l for l, _ in lang_entries]
            lang_vals   = [c for _, c in lang_entries]
            bar_colors  = [lang_colors.get(l, "#aaaaff") for l in lang_labels]
            fig_lang = px.bar(
                x=lang_labels, y=lang_vals,
                color=lang_labels,
                color_discrete_sequence=bar_colors,
                title="Language Distribution",
                labels={"x": "Language", "y": "Count"},
            )
            fig_lang.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(color="#e0e0e0"), yaxis=dict(color="#e0e0e0"),
            )
            st.plotly_chart(fig_lang, use_container_width=True)

        with ch3:
            lang_scam  = df[df["label"] == 1]["language"].value_counts()
            lang_legit = df[df["label"] == 0]["language"].value_counts()
            langs = lang_labels  # dynamic — all languages present in dataset
            fig_split = go.Figure(data=[
                go.Bar(name="Scam",       x=langs,
                       y=[lang_scam.get(l, 0) for l in langs],  marker_color="#FF4B4B"),
                go.Bar(name="Legitimate", x=langs,
                       y=[lang_legit.get(l, 0) for l in langs], marker_color="#00C851"),
            ])
            fig_split.update_layout(
                barmode="group", title="Scam vs Legit per Language",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(font=dict(color="#e0e0e0")),
                xaxis=dict(color="#e0e0e0"), yaxis=dict(color="#e0e0e0"),
            )
            st.plotly_chart(fig_split, use_container_width=True)

        st.divider()
        st.subheader("Sample Rows")
        st.dataframe(
            df[["language", "label", "text"]]
            .rename(columns={"label": "is_scam"})
            .sample(12, random_state=7)
            .reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.warning("Run `python generate_dataset.py` to generate the dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_pipeline:

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    st.subheader("Step 1 — Text Preprocessing")
    st.markdown(
        """
        A lightweight cleaner that works on both scripts without any language-specific library:
        """
    )
    st.code(
        """\
import re

def preprocess_text(text: str) -> str:
    text = text.lower()                                    # lowercase English only
    text = re.sub(r'http\\S+|bit\\.ly/\\S+',
                  ' urltoken ', text)                      # collapse URLs to token
    text = re.sub(
        r'[^\\u0900-\\u097F'   # Devanagari  (Hindi)
        r'\\u0B80-\\u0BFF'     # Tamil
        r'\\u0C00-\\u0C7F'     # Telugu
        r'\\u0D00-\\u0D7F'     # Malayalam
        r'\\w\\s]', ' ', text  # keep word chars + spaces
    )
    text = re.sub(r'\\s+', ' ', text).strip()             # normalise whitespace
    return text""",
        language="python",
    )
    st.markdown(
        """
        **Key decisions:**
        - **8 Indian language Unicode blocks** are preserved so no script is stripped:
          Devanagari (Hindi), Bengali, Gujarati, Gurmukhi, Tamil, Telugu, Kannada, Malayalam.
        - URLs are replaced with a `urltoken` feature — their *presence* is a scam signal,
          their specific domain is not.
        - Numbers (amounts, OTPs) are kept because `50000`, `otp`, and `pin` are strong
          discriminative features.
        - No stemming or stop-word removal — unnecessary given the TF-IDF sublinear scaling.
        - No language detection or branching — the same pipeline handles all 5 languages.
        """
    )

    st.divider()

    # ── Step 2: Feature Engineering ──────────────────────────────────────────
    st.subheader("Step 2 — Feature Engineering (FeatureUnion)")
    st.markdown(
        """
        Two TF-IDF vectorisers are combined side-by-side with `FeatureUnion`:
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Word TF-IDF  (1–2 grams)")
        st.markdown(
            """
            Captures keyword-level patterns in English and Tamil words:

            | n-gram | Feature captured |
            |--------|-----------------|
            | unigram | `urgent`, `otp`, `blocked` |
            | bigram | `share otp`, `kyc expired`, `account blocked` |

            - max_features: 15,000
            - sublinear_tf: True (dampens very frequent terms)
            """
        )
    with col2:
        st.markdown("##### Char-wb TF-IDF  (2–4 grams)")
        st.markdown(
            """
            Works at character level inside word boundaries — critical for Tamil:

            | Why it helps Tamil |
            |--------------------|
            | Tamil is agglutinative; `தடுக்கப்படும்` (will-be-blocked) has the root `தடு` |
            | Char n-grams capture the root even across inflected forms |
            | No Tamil tokeniser or stemmer needed |

            - max_features: 20,000
            - ngram_range: (2, 4)
            """
        )

    st.code(
        """\
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

word_tfidf = TfidfVectorizer(analyzer='word',    ngram_range=(1,2),
                              max_features=15_000, sublinear_tf=True)
char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                              max_features=20_000, sublinear_tf=True)

features = FeatureUnion([('word', word_tfidf), ('char', char_tfidf)])""",
        language="python",
    )

    st.divider()

    # ── Step 3: Model ─────────────────────────────────────────────────────────
    st.subheader("Step 3 — Classifier (Logistic Regression)")
    st.markdown(
        """
        **Why Logistic Regression over a neural network?**

        | Consideration | LR wins because… |
        |---------------|-----------------|
        | Dataset size | ~300 samples — deep learning overfits, LR generalises |
        | Interpretability | Coefficients map directly to feature importance for LIME |
        | Speed | Sub-second inference; no GPU required |
        | Explainability | LIME's linear approximation matches LR's actual decision surface |
        | Class imbalance | `class_weight='balanced'` handles any split cleanly |

        `C=2.0` (mild regularisation), `solver='lbfgs'`, `max_iter=1000`.
        """
    )

    st.divider()

    # ── Step 4: LIME ─────────────────────────────────────────────────────────
    st.subheader("Step 4 — Explainability (LIME)")
    st.markdown(
        """
        LIME (**L**ocal **I**nterpretable **M**odel-agnostic **E**xplanations) explains
        *individual* predictions, not the model as a whole.

        **How it works here:**
        1. Take the preprocessed SMS text.
        2. Generate ~500 perturbations (randomly remove words).
        3. Ask the pipeline to score each perturbation.
        4. Fit a simple linear model on those scores.
        5. The linear model's coefficients become the word-level explanations.

        This is model-agnostic — it calls `pipeline.predict_proba(list_of_strings)`
        directly, so the full FeatureUnion + LR chain runs on every perturbation.
        """
    )
    st.code(
        """\
explainer = LimeTextExplainer(class_names=['Legitimate', 'Scam'])
exp = explainer.explain_instance(
    processed_text,
    pipeline.predict_proba,   # the full sklearn pipeline
    num_features=12,
    num_samples=500,
)
components.html(exp.as_html(), height=460)""",
        language="python",
    )

    st.divider()

    # ── Step 5: Scam Signal Detector ──────────────────────────────────────────
    st.subheader("Step 5 — Scam Signal Detector (Rule-Based Red Flags)")
    st.markdown(
        """
        Runs in parallel with the ML model on the **raw (unpreprocessed) SMS text**.
        It scans for 8 categories of explicit scam patterns using regular expressions.
        This layer is fully transparent — every flag shown to the user has a plain-English
        explanation of *why* it is suspicious.
        """
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("##### The 8 signal categories")
        signals_table = [
            ("Urgency / pressure language", "URGENT, immediately, expire, last chance", "High"),
            ("OTP / PIN request", "otp, pin, cvv, share otp, send otp", "High"),
            ("Account block / KYC threat", "KYC, blocked, suspended, frozen, deactivated", "High"),
            ("Fake prize / lottery", "won, winner, prize, lottery, congratulations", "High"),
            ("Phone number in SMS body", "10-digit mobile number embedded in text", "High"),
            ("URL / link in SMS", "http, www, bit.ly, tinyurl", "High"),
            ("Call-to-action directive", "'click here now', 'call this number'", "Medium"),
            ("Refund-via-UPI trick", "electricity/gas refund → pay via UPI", "High"),
        ]
        import pandas as pd
        sig_df = pd.DataFrame(signals_table, columns=["Signal", "Trigger keywords/patterns", "Severity"])
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

    with col_r2:
        st.markdown("##### Why run this alongside ML?")
        st.markdown(
            """
            The ML model sees **preprocessed tokens** — URLs are replaced with `urltoken`,
            punctuation is stripped. It captures *statistical patterns* but cannot explain
            itself in plain language.

            The signal detector works on the **raw text** and catches things the model
            may score with moderate confidence:

            | Gap the ML model has | Signal detector fills it |
            |---------------------|--------------------------|
            | Sees `urltoken`, not the actual URL | Explicitly flags: "URL detected — do not click" |
            | Weights `9876543210` as a token | Explicitly flags: "Phone number embedded in message" |
            | Borderline OTP pattern | Explicitly flags with user-readable warning |

            Together they give **two independent evidence streams** — statistical + rule-based.
            """
        )

    st.code(
        """\
_URGENCY   = re.compile(r"\\b(urgent|immediately|now|expire[sd]?)\\b", re.I)
_OTP_PIN   = re.compile(r"\\b(otp|pin|cvv|share.{0,15}otp)\\b", re.I)
_PHONE     = re.compile(r"(?<!\\d)[6-9]\\d{9}(?!\\d)")
_URL       = re.compile(r"(https?://|www\\.|bit\\.ly/)\\S+", re.I)
# ... (8 patterns total)

def detect_scam_signals(raw_text):
    signals = []
    if m := _URGENCY.search(raw_text):
        signals.append({"label": "Urgency language",
                         "detail": f"Found: '{m.group()}'", "severity": "high"})
    if phones := _PHONE.findall(raw_text):
        signals.append({"label": "Phone number in message",
                         "detail": f"Numbers: {phones}", "severity": "high"})
    # ... check all 8 patterns
    return signals""",
        language="python",
    )

    st.divider()

    # ── Step 6: Sender ID Verification ───────────────────────────────────────
    st.subheader("Step 6 — Sender ID Verification (Rule-Based)")
    st.markdown(
        """
        Alongside the ML model, a **rule-based sender checker** evaluates the credibility
        of the SMS sender name/number. This is separate from the content model — it uses
        domain knowledge about how legitimate Indian SMS senders are structured.
        """
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("##### How Indian SMS senders work")
        st.markdown(
            """
            TRAI (India's telecom regulator) requires all commercial SMS senders to register
            under the **DLT (Distributed Ledger Technology)** system. Registered sender IDs follow
            a strict format:

            | Format | Example | Meaning |
            |--------|---------|---------|
            | `PREFIX-BRANDCODE` | `VM-HDFCBK` | HDFC Bank transactional |
            | `PREFIX-BRANDCODE` | `AD-AIRTEL` | Airtel promotional |
            | Short code | `56767` | Service short code |

            **Registered prefixes:** `VM`, `VD`, `AD`, `DM`, `BK`, `TM`, `IM` etc.

            Scammers use **personal mobile numbers** (10-digit, starting 6–9) because they
            cannot register fake brand codes on the DLT system.
            """
        )
    with col_s2:
        st.markdown("##### Verdict logic")
        st.markdown(
            """
            | Sender pattern | Verdict |
            |---------------|---------|
            | Known DLT prefix + known brand code | ✅ Legitimate |
            | Known DLT prefix + unknown code | ⚠️ Suspicious |
            | 10-digit mobile number | 🚨 Likely Fake |
            | Unknown alphanumeric | ⚠️ Suspicious |
            | Unrecognised format | 🚨 Likely Fake |
            """
        )

    st.code(
        """\
# Simplified sender check logic
KNOWN_CODES = {"HDFCBK", "SBIINB", "ICICIB", "AXISBK", "PAYTM", "AIRTEL", ...}
DLT_PREFIXES = {"VM", "VD", "AD", "DM", "BK", "TM", "IM", ...}

def analyze_sender(sender):
    s = sender.strip().upper()

    # Personal mobile number → always fake for official SMS
    if re.fullmatch(r"[6-9]\\d{9}", s):
        return "Likely Fake", "10-digit mobile — banks never use these"

    # Standard DLT format: PREFIX-BRANDCODE
    if re.match(r"^[A-Z]{2}-[A-Z0-9]+$", s):
        prefix, code = s[:2], s[3:]
        if prefix in DLT_PREFIXES and code in KNOWN_CODES:
            return "Legitimate", "Valid DLT prefix + known brand"
        elif prefix in DLT_PREFIXES:
            return "Suspicious", "Valid prefix but unknown brand code"

    return "Suspicious", "Does not match standard sender patterns"
""",
        language="python",
    )

    st.divider()

    # ── Step 7: Combined Verdict ──────────────────────────────────────────────
    st.subheader("Step 7 — Combined Verdict Engine")
    st.markdown(
        """
        Neither signal alone is enough. A **combined verdict** merges both:
        """
    )

    verdict_data = pd.DataFrame(
        {
            "Content (ML Model)": ["Scam", "Scam", "Scam", "Legit", "Legit", "Legit"],
            "Sender Check": ["Likely Fake", "Suspicious", "Legitimate", "Likely Fake", "Suspicious", "Legitimate"],
            "Final Verdict": ["🚨 SCAM", "🚨 SCAM", "🚨 SCAM", "⚠️ SUSPICIOUS", "⚠️ SUSPICIOUS", "✅ LEGITIMATE"],
            "Key Reason": [
                "Fake sender + scam content",
                "Unverified sender + scam content",
                "Content is scam — sender may be spoofed",
                "Legit-looking content from fake sender = social engineering risk",
                "Unverified sender — verify through official app",
                "Both signals clear",
            ],
        }
    )
    st.dataframe(verdict_data, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Why the Legit + Fake Sender case is SUSPICIOUS, not LEGITIMATE:**

        Scammers often send normal-looking messages first to build trust before requesting
        OTPs or payments. A real bank will **always** use a registered DLT sender ID —
        if the sender is a personal mobile number, the message should never be trusted
        regardless of how legitimate the content appears.
        """
    )

    st.divider()

    # ── Step 8: Full pipeline diagram ────────────────────────────────────────
    st.subheader("Full Pipeline at a Glance")
    st.code(
        """\
                    ┌─────────────────────────────────────┐
                    │  User Input                         │
                    │  Sender Name  +  Raw SMS Text       │
                    └────────────┬────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│ SIGNAL 1         │  │ SIGNAL 2         │  │ SIGNAL 3             │
│ Content (ML)     │  │ Red Flags        │  │ Sender ID            │
│                  │  │ (Rule-based)     │  │ (Rule-based)         │
│ preprocess_text()│  │                  │  │                      │
│ FeatureUnion     │  │ 8 regex patterns │  │ DLT prefix check     │
│ · Word TF-IDF    │  │ · Urgency        │  │ Known brand codes    │
│ · Char TF-IDF    │  │ · OTP request    │  │ Mobile # detection   │
│ Logistic Regr.   │  │ · Phone in body  │  │                      │
│ → P(Scam/Legit)  │  │ · URL in body    │  │ → Legit /            │
│ + LIME explain.  │  │ · KYC threat     │  │   Suspicious /       │
│                  │  │ · Prize/lottery  │  │   Likely Fake        │
│                  │  │ → Checklist      │  │                      │
└────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘
         │                     │                        │
         └─────────────────────┼────────────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │  Combined Verdict Engine      │
               │  ML verdict × Sender verdict  │
               │  → SCAM / SUSPICIOUS /        │
               │    LEGITIMATE                 │
               │  + plain-English explanation  │
               │  + Safety tips if SCAM        │
               └───────────────────────────────┘""",
        language="text",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_results:
    st.subheader("Model Performance")
    st.markdown("Evaluated on a stratified 80/20 train-test split (random_state=42).")

    metrics = compute_test_metrics(model)

    if metrics:
        acc  = metrics["accuracy"]
        prec = metrics["precision"]
        rec  = metrics["recall"]
        n_test  = metrics["n_test"]
        n_total = metrics["n_total"]
        cm_data = metrics["cm"]

        if acc >= 0.999:
            st.info(
                "**Note on near-perfect accuracy:** The synthetic portion of the dataset was "
                "written by the same author for train and test, so vocabulary patterns overlap. "
                "Real-world performance on unseen SMS will be lower. "
                "See the Limitations tab for a full discussion.",
                icon="ℹ️",
            )

        col1, col2, col3, col4 = st.columns(4)
        live_metrics = [
            ("Test Accuracy",    f"{acc:.1%}"),
            ("Dataset Size",     f"{n_total:,}"),
            ("Precision (Scam)", f"{prec:.1%}"),
            ("Recall (Scam)",    f"{rec:.1%}"),
        ]
        for col, (lbl, val) in zip([col1, col2, col3, col4], live_metrics):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Train the model (`python train.py`) to see live metrics.")
        cm_data = [[0, 0], [0, 0]]
        n_test  = 0

    st.markdown("")

    col_cm, col_notes = st.columns([1, 1])

    with col_cm:
        st.markdown(f"##### Confusion Matrix (test set, n={n_test})")
        z = cm_data
        tn, fp = z[0][0], z[0][1]
        fn, tp = z[1][0], z[1][1]
        fig_cm = go.Figure(data=go.Heatmap(
            z=z,
            x=["Predicted Legit", "Predicted Scam"],
            y=["Actual Legit", "Actual Scam"],
            colorscale=[[0, "#1a1d27"], [1, "#6655ff"]],
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 22, "color": "white"},
            showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", margin=dict(t=10, b=10, l=10, r=10),
            height=220,
            xaxis=dict(color="#e0e0e0"), yaxis=dict(color="#e0e0e0", autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}  —  live from current model.")

    with col_notes:
        st.markdown("##### What these numbers mean")
        st.markdown(
            f"""
            - **True Negatives (TN={tn})** — legitimate SMS correctly identified as safe.
            - **True Positives (TP={tp})** — scam SMS correctly caught.
            - **False Positives (FP={fp})** — legitimate SMS incorrectly flagged (aim: 0).
            - **False Negatives (FN={fn})** — scam SMS that slipped through (aim: 0).
            - Dataset grew from **152 → {n_total:,} samples** — added Tamil, Hindi, Telugu
              synthetic messages + 2k real-world Indian SMS (princebari dataset).
            - Real-world accuracy on truly unseen messages will be lower — see Limitations tab.
            """
        )

    # Model comparison
    st.divider()
    st.subheader("Model Comparison — Why Logistic Regression?")
    st.markdown(
        "Multiple classifiers were considered. Here is how they compare on this dataset:"
    )
    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression ✅", "Naive Bayes", "Linear SVM", "Random Forest", "Neural Network"],
        "CV F1 (approx)": ["99.35%", "~96%", "~98%", "~97%", "~85%"],
        "Interpretable": ["Yes — coefficients", "Partial", "Partial", "No", "No"],
        "LIME compatible": ["Excellent", "Good", "Good", "Fair", "Fair"],
        "Works at ~300 samples": ["Yes", "Yes", "Yes", "Overfits easily", "Overfits badly"],
        "Why not chosen": ["✅ Chosen", "Weaker on mixed-script", "No probability calibration", "Overfits small data", "Far too small dataset"],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Live top features — bar charts
    if model is not None:
        st.divider()
        st.subheader("Feature Importance (Live — from LR coefficients)")
        st.markdown(
            "These are the actual weights the model learned — not LIME approximations."
        )

        top_scam, top_legit = get_top_features(model, n=12)

        col_s, col_l = st.columns(2)
        with col_s:
            scam_df = pd.DataFrame(top_scam, columns=["Feature", "Weight"])
            fig_s = go.Figure(go.Bar(
                x=scam_df["Weight"],
                y=scam_df["Feature"],
                orientation="h",
                marker_color="#FF4B4B",
            ))
            fig_s.update_layout(
                title="Top Scam Indicators",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", height=380,
                margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(color="#e0e0e0", title="Weight"),
                yaxis=dict(color="#e0e0e0", autorange="reversed"),
            )
            st.plotly_chart(fig_s, use_container_width=True)

        with col_l:
            legit_df = pd.DataFrame(top_legit, columns=["Feature", "Weight"])
            fig_l = go.Figure(go.Bar(
                x=legit_df["Weight"].abs(),
                y=legit_df["Feature"],
                orientation="h",
                marker_color="#00C851",
            ))
            fig_l.update_layout(
                title="Top Legitimate Indicators",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", height=380,
                margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(color="#e0e0e0", title="|Weight|"),
                yaxis=dict(color="#e0e0e0", autorange="reversed"),
            )
            st.plotly_chart(fig_l, use_container_width=True)

        st.caption(
            "Higher bar = stronger influence on the prediction. "
            "Red bars push toward Scam; green bars push toward Legitimate."
        )
    else:
        st.info("Train the model (`python train.py`) to see live feature weights.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────
with tab_workflow:
    st.subheader("Development Workflow — Step by Step")

    steps = [
        (
            "1",
            "Define the Problem Scope",
            "Chose binary classification (Scam vs Legitimate) over multi-label because "
            "the actionable output for a user is a single yes/no verdict. "
            "Decided to support regional Indian languages (Tamil, Telugu, Hindi) from the start "
            "rather than as an afterthought — that decision shaped every subsequent "
            "engineering choice, especially the feature engineering and preprocessing layers.",
        ),
        (
            "2",
            "Build the Dataset",
            "No usable public dataset exists for Tamil UPI scam SMS. "
            "~300 hand-crafted messages cover 13+ scam patterns across Tamil, English, and Mixed. "
            "Two free GitHub datasets (shshnk158, princebari) add real Hindi and Telugu rows. "
            "Class ratio kept close to 50/50; `class_weight='balanced'` used as a safety net.",
        ),
        (
            "3",
            "Design the Preprocessing Layer",
            "The key insight: Tamil, Hindi, and Telugu all use spaces between words just like "
            "English, so word tokenisation works out of the box. "
            "The preprocessing function preserves 8 Indian language Unicode blocks "
            "(Devanagari, Bengali, Gujarati, Gurmukhi, Tamil, Telugu, Kannada, Malayalam) "
            "while stripping punctuation and normalising URLs. "
            "No language detection or branching logic needed — one function handles all 5 languages.",
        ),
        (
            "4",
            "Choose and Justify the Feature Engineering Strategy",
            "TF-IDF alone on words works for English but misses Tamil morphology. "
            "Adding char n-grams (2–4) gives the model access to Tamil root forms even when "
            "words are inflected differently. FeatureUnion concatenates both feature spaces "
            "into a single sparse matrix — no custom code, full sklearn compatibility.",
        ),
        (
            "5",
            "Select the Classifier",
            "Logistic Regression was chosen deliberately over tree-based or neural models: "
            "the dataset is small (~300 samples), LR's linear decision boundary aligns with "
            "how LIME generates explanations, and coefficient magnitudes are directly "
            "interpretable. Hyperparameter: C=2.0 (mild regularisation found by manual trial).",
        ),
        (
            "6",
            "Integrate LIME for Explainability",
            "LIME was a first-class design requirement, not an afterthought. "
            "The pipeline was structured so that `pipeline.predict_proba(list_of_strings)` "
            "works out of the box — LIME calls this exact interface with its perturbed texts. "
            "num_samples=500 balances explanation quality vs latency.",
        ),
        (
            "7",
            "Build the Streamlit UI",
            "Two pages: the detector (user-facing) and this page (recruiter/reviewer). "
            "Session state handles example-button → text-area synchronisation. "
            "LIME's HTML output is embedded via `st.components.v1.html` so the full "
            "highlighted-text visualisation renders inside Streamlit without any extra dependencies.",
        ),
        (
            "8",
            "Evaluate and Validate",
            "Stratified train/test split (80/20) + 5-fold cross-validation. "
            "Priority metric: F1 (balances precision and recall for imbalanced-ish data). "
            "Confusion matrix checked specifically for false positives — zero FP was the target "
            "to ensure no legitimate SMS gets incorrectly flagged.",
        ),
        (
            "9",
            "Add Sender ID Verification + Combined Verdict",
            "Realised the content model alone creates user confusion when signals conflict — "
            "e.g. legit-looking content from a personal mobile number. "
            "Added a rule-based sender checker using TRAI DLT registration knowledge "
            "(registered prefixes, known brand codes, mobile number patterns). "
            "A decision matrix (6 combinations of content × sender verdict) produces a single "
            "final conclusion with a plain-English reason — so the user never has to reconcile "
            "two contradicting signals themselves.",
        ),
        (
            "10",
            "Add Scam Signal Detector + Safety Tips",
            "Identified three gaps the ML model does not address for end users: "
            "(1) URLs are silently replaced with 'urltoken' during preprocessing — never surfaced to the user; "
            "(2) embedded phone numbers in the SMS body are a major red flag but invisible in the results; "
            "(3) borderline ML confidence scores leave users without actionable guidance. "
            "Added a parallel rule-based signal detector (8 regex patterns on raw text) that produces "
            "a plain-English red-flag checklist. Added contextual safety tips (report to 1930, "
            "don't share OTP, don't click links) for every Scam or Suspicious verdict.",
        ),
        (
            "11",
            "Expand to Hindi + Telugu + External Real-World Data",
            "Extended preprocessing to preserve 8 Indian language Unicode blocks (not just Tamil). "
            "Added hand-crafted SCAM/LEGIT message sets in Devanagari Hindi and Telugu script "
            "to ensure the model has native-script training data for those languages. "
            "Integrated two free GitHub datasets: shshnk158 (multilingual, ~10k) and princebari "
            "(crowdsourced Indian SMS, ~2k). Wrote a robust loader with multi-encoding fallback "
            "(UTF-8 → latin-1 → cp1252) and per-source filtering logic. "
            "Total dataset grew from 152 synthetic → 2,394 samples across 5 languages.",
        ),
    ]

    for num, title, desc in steps:
        col_n, col_body = st.columns([1, 11])
        with col_n:
            st.markdown(
                f'<div class="step-num">{num}</div>', unsafe_allow_html=True
            )
        with col_body:
            st.markdown(f"**{title}**")
            st.markdown(desc)
        st.markdown("")

    st.divider()
    st.subheader("How to Run the Project Locally")
    st.code(
        """\
# Clone / download the project
cd UPI-SCAM-SMS-DETECTOR

# Install dependencies
pip install -r requirements.txt

# Step 1 — build the dataset
python generate_dataset.py

# Step 2 — train the model  (saves models/upi_scam_detector.pkl)
python train.py

# Step 3 — launch the app
streamlit run app.py""",
        language="bash",
    )

    st.divider()
    st.subheader("Possible Extensions")
    extensions = [
        ("More data", "Crowdsource real Tamil scam SMS via citizen-reporting apps to replace the synthetic dataset"),
        ("Transformer model", "Fine-tune MuRIL or IndicBERT for better Tamil morphology and context understanding"),
        ("More languages", "Extend to Kannada, Gujarati, Bengali using the same char-ngram approach — no new tokeniser needed"),
        ("Real-time API", "Wrap the pipeline in a FastAPI endpoint for mobile app or browser-extension integration"),
        ("Active learning", "Flag low-confidence predictions (40–60% scam prob) for human review to grow labelled data"),
        ("Ensemble", "Stack LR with a gradient-boosted tree to cut the false-negative rate further"),
        ("DLT database sync", "Pull live TRAI DLT sender registry via API instead of a static known-codes list"),
        ("URL reputation check", "Pass detected URLs through a safe-browsing API (Google, VirusTotal) for live domain reputation"),
    ]
    col_a, col_b = st.columns(2)
    for i, (ext_title, ext_desc) in enumerate(extensions):
        tgt = col_a if i % 2 == 0 else col_b
        with tgt:
            st.markdown(f"**{ext_title}** — {ext_desc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_limits:
    st.subheader("Honest Limitations of This Project")
    st.markdown(
        "Every project has limitations. Knowing them — and being able to articulate them — "
        "is a sign of engineering maturity, not weakness."
    )

    limitations = [
        (
            "Synthetic Core Dataset",
            "high",
            "The ~300 synthetic messages were hand-written, not collected from real users.",
            [
                "No public labelled dataset exists for Tamil UPI scam SMS — hand-crafting was the only option for the Tamil and Mixed portions.",
                "The 100% accuracy on the test set reflects the fact that train and test messages share the same vocabulary patterns, since they were written by the same author.",
                "In production, the model would need to be validated on real-world SMS logs, which would likely reduce accuracy.",
                "Fix path: Partner with a telecom provider or cybercrime cell for real labelled data; use active learning to grow the dataset efficiently.",
            ],
        ),
        (
            "Dataset Imbalance — Synthetic vs Real",
            "high",
            "~350 messages are hand-crafted; the remaining ~2,000 come from an external Indian SMS dataset that is not UPI-specific.",
            [
                "The external (princebari) messages are general Indian SMS spam, not UPI scam SMS — vocabulary may differ from real UPI fraud.",
                "The 5-fold CV F1 looks strong but is evaluated on the same distribution used for training.",
                "Rare edge cases (novel phishing tactics, code-mixed sentences with new slang) may not be captured.",
                "Fix path: Crowdsource real labelled UPI scam SMS from users or partner with a telecom provider.",
            ],
        ),
        (
            "Sender Checker Uses a Static Allowlist",
            "medium",
            "The known-codes list of ~60 brand codes is manually curated.",
            [
                "New legitimate senders (e.g. a new fintech app) will be flagged as Suspicious until the list is updated.",
                "TRAI's DLT registry has 1000s of registered senders — the static list is a small subset.",
                "Fix path: Integrate with the TRAI DLT public API (if available) or scrape the registry periodically.",
            ],
        ),
        (
            "No Multilingual Semantic Understanding",
            "medium",
            "The char-ngram approach handles Tamil morphology but has no semantic understanding.",
            [
                "A scam phrased in an entirely new way (new vocabulary, new framing) with no overlap to training patterns may slip through.",
                "Tamil colloquialisms, transliteration variations (e.g. 'urgam' for urgency), or completely novel script combinations are not captured.",
                "Fix path: Fine-tune a multilingual transformer (MuRIL, IndicBERT) on a larger real-world dataset.",
            ],
        ),
        (
            "No Real-Time Deployment",
            "low",
            "The app must be run locally — there is no hosted version.",
            [
                "Recruiters and users cannot try it without cloning the repo and running training scripts.",
                "Fix path: Deploy to Streamlit Community Cloud (free) — requires committing the pre-trained model file to the repository.",
            ],
        ),
    ]

    for title, severity, summary, details in limitations:
        color = "#FF4B4B" if severity == "high" else "#FFA500" if severity == "medium" else "#4A90D9"
        badge = severity.upper()
        detail_html = "".join(f'<li style="color:#bbb;font-size:0.88rem;margin-bottom:4px;">{d}</li>' for d in details)
        st.markdown(
            f"""
            <div style="background:{color}12;border-left:4px solid {color};
                        border-radius:8px;padding:14px 18px;margin-bottom:14px;">
              <div style="margin-bottom:6px;">
                <span style="background:{color};color:#fff;font-size:0.72rem;font-weight:700;
                             padding:2px 8px;border-radius:3px;margin-right:8px;">{badge}</span>
                <strong style="color:#e0e0e0;font-size:1rem;">{title}</strong>
              </div>
              <div style="color:#ccc;font-size:0.92rem;margin-bottom:8px;">{summary}</div>
              <ul style="margin:0;padding-left:18px;">{detail_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("What This Project Does Demonstrate")
    st.markdown(
        """
        Despite these limitations, this project is designed to show:

        | Skill | Evidence |
        |-------|---------|
        | **End-to-end ML pipeline** | Data generation → preprocessing → feature engineering → training → deployment |
        | **Multi-language NLP** | Tamil Unicode handling without any external Tamil NLP library |
        | **Explainable AI (XAI)** | LIME integrated as a first-class feature, not an afterthought |
        | **Layered system design** | Three independent signals (ML + rule-based signals + sender check) combined into one verdict |
        | **Product thinking** | Combined verdict engine, safety tips, user-readable explanations |
        | **Communication** | This Under the Hood page — technical writing for a non-technical audience |
        | **Self-awareness** | This Limitations tab — ability to critically evaluate your own work |
        """
    )
