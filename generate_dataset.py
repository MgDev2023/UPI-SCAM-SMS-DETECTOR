"""
Generate the labeled SMS dataset.
Combines:
  1. Hand-crafted synthetic UPI-specific messages (Tamil, English, Mixed)
  2. shshnk158 multilingual dataset — real Hindi + Telugu + English spam/ham (~10k)
  3. princebari Indian dataset  — real Hindi + English spam/ham (~2k)

Run once: python generate_dataset.py
Output  : data/dataset.csv
"""

import os
import re
import pandas as pd


# ── Language auto-detection ───────────────────────────────────────────────────
def _detect_lang(text: str) -> str:
    t = str(text)
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", t))
    has_telugu     = bool(re.search(r"[\u0C00-\u0C7F]", t))
    has_tamil      = bool(re.search(r"[\u0B80-\u0BFF]", t))
    has_latin      = bool(re.search(r"[a-zA-Z]", t))

    if has_devanagari and has_latin:
        return "Mixed-Hindi"
    if has_devanagari:
        return "Hindi"
    if has_telugu and has_latin:
        return "Mixed-Telugu"
    if has_telugu:
        return "Telugu"
    if has_tamil and has_latin:
        return "Mixed"
    if has_tamil:
        return "Tamil"
    return "English"


# ── External dataset loader ───────────────────────────────────────────────────
_SOURCES = [
    {
        "name": "shshnk158-multilingual",
        "url": (
            "https://raw.githubusercontent.com/shshnk158/"
            "Multilingual-SMS-spam-detection-using-RNN/master/Resources/spam.csv"
        ),
        "text_col": "msg",
        "label_col": "label",
        "label_map": {"ham": 0, "spam": 1},
        # Keep only Unicode-script Hindi/Telugu rows (this dataset has real Devanagari/Telugu)
        "filter_indian_only": True,
    },
    {
        "name": "princebari-indian",
        "url": (
            "https://raw.githubusercontent.com/princebari/"
            "-SMS-Spam-Classification-on-Indian-Dataset-"
            "A-Crowdsourced-Collection-of-Hindi-and-English-Messages"
            "/main/indian_spam.csv"
        ),
        "text_col": "v2",
        "label_col": "v1",
        "label_map": {"ham": 0, "spam": 1},
        # This dataset uses Roman-script Hinglish — no Devanagari Unicode detected.
        # Keep ALL rows: it is crowdsourced from 43 Indian participants, so even the
        # "English" messages are India-specific SMS context (not generic UCI English spam).
        "filter_indian_only": False,
    },
]

# Encodings to try in order for each source
_ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]


def load_external_datasets() -> pd.DataFrame:
    """Download free Hindi/Telugu SMS spam datasets and normalise to (text, label, language)."""
    frames = []
    for src in _SOURCES:
        try:
            print(f"  Downloading {src['name']} ...")

            # Try multiple encodings — some CSVs are latin-1, not UTF-8
            df = None
            for enc in _ENCODINGS:
                try:
                    df = pd.read_csv(src["url"], encoding=enc, on_bad_lines="skip")
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if df is None:
                raise ValueError("All encodings failed — could not read the CSV.")

            # Flexible column resolution (case-insensitive)
            tcol = src["text_col"]
            lcol = src["label_col"]
            if tcol not in df.columns or lcol not in df.columns:
                col_map = {c.lower(): c for c in df.columns}
                tcol = col_map.get(tcol.lower(), tcol)
                lcol = col_map.get(lcol.lower(), lcol)

            df = df[[tcol, lcol]].dropna()
            df.columns = ["text", "label_raw"]
            df["text"]  = df["text"].astype(str).str.strip()
            df["label"] = df["label_raw"].str.strip().str.lower().map(src["label_map"])
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)

            # Detect language per row
            df["language"] = df["text"].apply(_detect_lang)

            if src["filter_indian_only"]:
                # Keep only rows with Devanagari / Telugu Unicode (real script, not Romanised)
                keep = df["language"].isin(["Hindi", "Telugu", "Mixed-Hindi", "Mixed-Telugu"])
                df = df[keep]
                print(f"    → {len(df)} Hindi/Telugu (Unicode script) rows kept from {src['name']}")
            else:
                # Keep all rows — dataset is India-specific regardless of script
                print(f"    → {len(df)} rows kept from {src['name']} (all Indian SMS context)")

            print(f"      {df['language'].value_counts().to_dict()}")
            frames.append(df[["text", "label", "language"]])

        except Exception as exc:
            print(f"  WARNING: Could not load {src['name']}: {exc}")

    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset="text")
    return pd.DataFrame(columns=["text", "label", "language"])

# ── Scam messages ─────────────────────────────────────────────────────────────

SCAM_ENGLISH = [
    # KYC / Account block threats
    "URGENT: Your UPI account will be BLOCKED! Update KYC immediately. Call 9876543210 now!",
    "Your KYC is expired. Paytm account will be suspended in 2 hours. Call now to update KYC",
    "Dear user your PhonePe KYC is incomplete. Account will be blocked. Update now at bit.ly/kyc",
    "NPCI ALERT: Your UPI ID will be deactivated. Complete mandatory update within 24 hours now",
    "FINAL WARNING: Failure to update KYC in 24 hours will result in permanent UPI account blocking",
    "Your Gpay account balance will be forfeited if KYC is not updated by tomorrow. Act fast now",
    "Dear customer your SBI account will be closed due to incomplete KYC. Share details to reactivate",
    "URGENT NOTICE: Your UPI is blocked due to KYC mismatch. Call 9988001122 immediately to resolve",
    "Your Aadhaar-linked UPI KYC has expired. Account will freeze tonight. Update PIN to continue",
    "SBI: KYC verification pending. Account restricted. Visit bit.ly/sbi-kyc or call 8899001122",
    # OTP / PIN requests
    "SBI ALERT: Account will be deactivated in 24 hours. Share your OTP with our agent to reactivate",
    "FREE OFFER: Get Rs 2000 in your PhonePe wallet. Just share your UPI PIN to activate this offer",
    "HDFC Bank: Suspicious activity on your account. Confirm your UPI PIN to secure your account now",
    "BANK ALERT: New device login detected. Share OTP to confirm it was you or your account is blocked",
    "JUSPAY: Unusual transaction detected on UPI. Reply with OTP to cancel the unauthorized payment",
    "Bank security team: Suspicious UPI request detected. Share your 6-digit PIN to block it now",
    "SBI customer your UPI is about to be disabled. Share PIN to continue using UPI services",
    "Your account has been compromised. Call our helpline immediately and share OTP to secure funds",
    "ICICI Bank: Fraud alert on your account. Verify identity by sharing your UPI PIN immediately",
    "Customer your UPI is compromised. Freeze account now by calling 8800998877 and sharing PIN",
    "UPI payment fraud detected from your account. Immediately share OTP to stop this transaction",
    # Prize / lottery
    "Congratulations! You won Rs 50000 in PhonePe Lucky Draw. Send OTP 5534 to claim your prize now",
    "Lottery Winner! Your number won Rs 25 lakhs. Pay processing fee of Rs 500 via UPI to claim now",
    "Amazon UPI offer: You won voucher worth Rs 5000. Share bank OTP to credit amount to your account",
    "WINNER: Monthly UPI cashback draw winner. Rs 50000 prize. OTP required to transfer the funds",
    "You have won a lucky prize in UPI diwali offer. Share OTP 7652 to claim your Rs 5000 reward",
    "You are selected as the winner of PhonePe anniversary lottery. Claim Rs 1 lakh now with OTP",
    "Congratulations you have been selected for UPI rewards program. Confirm your PIN to enroll now",
    "You won a reward on your recent UPI payment. Click the link to claim Rs 10000 cashback now",
    "REWARD: You are selected for Rs 10000 cashback on UPI. Enter OTP immediately to activate reward",
    # Government / fake schemes
    "Income Tax Dept: Rs 15000 tax refund pending. Provide your UPI ID and OTP for immediate credit",
    "PM Kisan Yojana: Rs 6000 ready for credit. Share your UPI PIN to receive government funds today",
    "Government COVID relief fund Rs 3500 approved for you. Share UPI ID with OTP to receive today",
    "RBI notification: New UPI guidelines require re-verification. Share account details now urgently",
    "PM Modi scheme: Rs 15000 for every UPI user. Limited time offer. Share PIN to receive funds",
    "PM Awas Yojana: Rs 2.5 lakh housing subsidy approved. Send UPI PIN to release the funds today",
    "EPFO: PF withdrawal of Rs 50000 ready. Confirm UPI PIN to transfer amount to your bank now",
    "Govt Skill India: Rs 8000 training stipend approved. Share UPI ID to receive payment today",
    "Aadhaar-linked subsidy Rs 1200 credited. Confirm with OTP to release funds to your account",
    "Election Commission: Voter ID linking bonus Rs 500. Share UPI PIN to receive the incentive now",
    # Suspicious activity / fraud alerts
    "Your Google Pay account is suspended due to suspicious activity. Verify now at gpay-verify.com",
    "ALERT: Unauthorized login detected on your UPI account. Share OTP immediately to block access",
    "Your UPI transaction of Rs 50000 is pending verification. Confirm OTP urgently to release funds",
    "Your transaction to unknown account has been initiated. Share OTP immediately to cancel it now",
    "URGENT NOTICE: Rs 10000 reversal pending on your account. Confirm OTP to receive the amount",
    # Refund scams
    "Electricity bill refund Rs 2000 via UPI. Enter UPI PIN code to receive the immediate credit",
    "Flipkart cashback Rs 1500 ready. Send your UPI ID and last 4 digits of card to receive it",
    "Bank of India: Rs 5000 cashback offer for loyal customers. Share your UPI PIN to receive it",
    "Gas cylinder subsidy Rs 300 approved by government. Share UPI PIN to receive the refund now",
    "Water board refund Rs 450 ready for your account. Enter OTP to receive the amount immediately",
    "LPG subsidy Rs 250 pending credit. Verify your UPI PIN with our agent to release the funds",
    # Job / loan scams
    "Work from home job offer: Earn Rs 15000/day. Pay Rs 500 registration fee via UPI to start",
    "Easy loan Rs 5 lakh approved for you. Pay processing fee Rs 999 via UPI. Call 9090909090",
    "Data entry job: Rs 800 per hour from home. Send Rs 299 via UPI for training kit access",
    "Part time job offer: Rs 50000/month. Pay Rs 1000 registration fee at paytm UPI immediately",
    "Instant personal loan Rs 2 lakh. No documents. Pay Rs 500 insurance fee via UPI to disburse",
    # Courier / delivery scams
    "Your parcel is held at customs. Pay Rs 199 via UPI to release the delivery immediately",
    "You won iPhone 14 in UPI Lucky Draw! Pay Rs 199 shipping fee via UPI to receive your prize",
    "Amazon delivery pending: Pay Rs 49 handling fee via UPI to reschedule your package delivery",
    "FedEx: Package held. Pay customs duty Rs 399 via UPI within 24 hours to avoid return",
    "India Post: Undelivered parcel. Pay Rs 29 redelivery fee via UPI: bit.ly/indiapost-pay",
    # SIM / Telecom scams
    "Your SIM will be blocked in 24 hours. Share Aadhaar OTP to complete mandatory re-verification",
    "Airtel: Your number 9876543210 will be deactivated. Share UPI PIN to pay outstanding dues now",
    "Jio alert: Your SIM card KYC expired. Call 8899776655 and share OTP to prevent disconnection",
    # Misc
    "Your Paytm account is about to expire. Update KYC now by calling 1800-XXX-XXXX immediately",
    "FREE cash offer: Rs 5000 added to your UPI. Enter OTP to unlock and transfer to your bank account",
    "Axis Bank: Your UPI limit has been upgraded to 2 lakhs. Confirm upgrade by sharing your OTP",
    "ICICI: Your account shows suspicious login from Delhi. Share PIN to block unauthorized access",
    "PhonePe: Rs 10000 bonus added to your account. Verify OTP at bit.ly/phonepe-bonus to claim",
    "Crypto investment: Turn Rs 1000 into Rs 50000 in 7 days. Send via UPI to wallet ID given",
    "Insurance claim Rs 75000 approved. Pay stamp duty Rs 499 via UPI to release the cheque now",
    "Court notice: Outstanding dues Rs 2300 on your UPI. Pay immediately or account will be seized",
    "Your mutual fund dividend Rs 3200 is ready. Share UPI PIN to transfer to your bank account",
    "RBI: UPI transaction limit increased. Verify by sharing OTP with our banking partner now",
]

SCAM_TAMIL = [
    # KYC / Account block threats
    "அவசரம்: உங்கள் UPI கணக்கு தடுக்கப்படும். KYC புதுப்பிக்க இப்போதே அழைக்கவும் 9876543210",
    "உங்கள் Paytm KYC காலாவதியானது. கணக்கு நிறுத்தப்படும். இப்போதே புதுப்பிக்கவும்",
    "Google Pay KYC முடிக்கப்படவில்லை. நாளை கணக்கு தடுக்கப்படும். இப்போதே புதுப்பிக்கவும்",
    "NPCI எச்சரிக்கை: உங்கள் UPI ID நீக்கப்படும். 24 மணி நேரத்தில் புதுப்பிக்கவும்",
    "உங்கள் SBI கணக்கு முழுமையற்ற KYC காரணமாக மூடப்படும். விவரங்கள் பகிர்ந்து மீட்டெடுக்கவும்",
    "ஆதார் இணைக்கப்பட்ட UPI KYC காலாவதி. 9988776655 அழைத்து மீண்டும் செயல்படுத்தவும்",
    "அவசரம்: 24 மணி நேரத்தில் KYC புதுப்பிக்காவிட்டால் உங்கள் PhonePe கணக்கு முடக்கப்படும்",
    "HDFC வங்கி: உங்கள் UPI KYC சரிபார்ப்பு நிலுவையில் உள்ளது. bit.ly/hdfc-kyc-ல் புதுப்பிக்கவும்",
    "உங்கள் Gpay கணக்கு இரவு தடுக்கப்படும். KYC புதுப்பிக்க 9900112233 அழைக்கவும் இப்போதே",
    "SBI: KYC சரிபார்ப்பு நிலுவை. கணக்கு கட்டுப்படுத்தப்பட்டது. bit.ly/sbi-verify ல் செய்யவும்",
    # OTP / PIN requests
    "SBI எச்சரிக்கை: 24 மணி நேரத்தில் கணக்கு நிறுத்தப்படும். OTP பகிர்ந்து மீண்டும் செயல்படுத்தவும்",
    "அரசு திட்டம்: Rs 5000 உதவித்தொகை அங்கீகரிக்கப்பட்டது. UPI PIN தரவும் பணம் பெற",
    "HDFC வங்கி: சந்தேகமான செயல்பாடு. UPI PIN பகிர்ந்து கணக்கை பாதுகாக்கவும் இப்போதே",
    "வங்கி பாதுகாப்பு குழு: சந்தேகமான UPI கோரிக்கை. PIN பகிர்ந்து தடுக்கவும் இப்போதே",
    "உங்கள் கணக்கு சமரசம் செய்யப்பட்டது. OTP பகிர்ந்து பணத்தை பாதுகாக்கவும் உடனே",
    "இலவச சலுகை: Rs 2000 PhonePe வாலட்டில். UPI PIN பகிர்ந்து செயல்படுத்தவும் இப்போதே",
    "Axis வங்கி: UPI வரம்பு 2 லட்சமாக உயர்த்தப்பட்டது. OTP பகிர்ந்து உறுதிப்படுத்தவும்",
    "உங்கள் UPI பரிமாற்றம் சரிபார்க்கப்படுகிறது. OTP உறுதிப்படுத்தவும் நிதி வெளியிட",
    "ICICI வங்கி: முழுமையற்ற சரிபார்ப்பு. உங்கள் 6 இலக்க PIN பகிரவும் கணக்கு திறக்க",
    "RBI: புதிய UPI வழிகாட்டுதல்கள். கணக்கு மீண்டும் சரிபார்க்க OTP பகிரவும் இப்போதே",
    # Prize / lottery
    "வாழ்த்துக்கள்! நீங்கள் Rs 50000 பரிசு வென்றீர்கள். OTP அனுப்பவும் 5534 கோரிக்கை செய்ய",
    "பரிசு ஒதுக்கீடு: UPI லக்கி டிரா Rs 25000 வென்றீர்கள். OTP உறுதிப்படுத்தவும் பெற",
    "லாட்டரி வெற்றி! Rs 25 லட்சம் வென்றீர்கள். Rs 500 செலுத்தி பரிசு பெறுங்கள் இப்போது",
    "Diwali சலுகை: UPI வழியாக Rs 5000 கேஷ்பேக் வென்றீர்கள். OTP 7652 அனுப்பவும் பெற",
    "PhonePe ஆண்டு விழா லாட்டரி வெற்றியாளராக நீங்கள் தேர்ந்தெடுக்கப்பட்டீர்கள். OTP அனுப்பவும்",
    "Amazon சலுகை: Rs 5000 வவுச்சர் வென்றீர்கள். வங்கி OTP பகிர்ந்து கணக்கில் பெறுங்கள்",
    "iPhone 14 வென்றீர்கள்! Rs 199 ஷிப்பிங் கட்டணம் UPI மூலம் செலுத்தவும் பரிசு பெற",
    "WINNER: உங்கள் UPI எண் Rs 10 லட்ச பரிசை வென்றது. OTP கொடுங்கள் பணம் பெற இப்போதே",
    "வாழ்த்துக்கள்! Flipkart பண்டிகை கேஷ்பேக் Rs 3000 வென்றீர்கள். PIN பகிரவும் பெற",
    "UPI Diwali draw: நீங்கள் Rs 50000 வென்றீர்கள். Rs 199 கட்டணம் செலுத்தி பரிசு பெறவும்",
    # Government / fake schemes
    "PM கிசான் திட்டம்: Rs 6000 அனுப்ப தயார். UPI PIN கொடுங்கள் அரசு பணம் பெற",
    "வருமான வரி திணைக்களம்: Rs 12000 திரும்பப் பெறல். UPI ID மற்றும் OTP கொடுங்கள்",
    "COVID நிவாரண நிதி Rs 3500 அங்கீகரிக்கப்பட்டது. UPI ID மற்றும் OTP கொடுங்கள்",
    "PM Awas Yojana: Rs 2.5 லட்ச வீட்டு மானியம் அங்கீகரிக்கப்பட்டது. UPI PIN கொடுங்கள் இப்போதே",
    "EPFO: PF திரும்பப் பெறல் Rs 50000 தயார். UPI PIN கொடுங்கள் வங்கிக்கு மாற்ற இன்றே",
    "அரசு திறன் இந்தியா: Rs 8000 பயிற்சி உதவித்தொகை. UPI ID பகிரவும் இன்றே பெற",
    "ஆதார் இணைக்கப்பட்ட மானியம் Rs 1200 வரவு. OTP கொடுங்கள் நிதி வெளியிட",
    "தேர்தல் ஆணையம்: வாக்காளர் அட்டை இணைப்பு போனஸ் Rs 500. UPI PIN கொடுங்கள் இப்போதே",
    "Aadhaar subsidy Rs 600 நிலுவையில் உள்ளது. OTP பகிர்ந்து உங்கள் கணக்கில் பெறவும்",
    # Refund scams
    "மின்சாரக் கட்டண திரும்பப் பெறல் Rs 2000 UPI மூலம். PIN உள்ளிடவும் பெற",
    "கேஸ் சிலிண்டர் மானியம் Rs 300 அரசால் அங்கீகரிக்கப்பட்டது. UPI PIN கொடுங்கள் பெற",
    "தண்ணீர் வாரிய திரும்பப் பெறல் Rs 450 தயார். OTP கொடுங்கள் உடனே பெற",
    "LPG மானியம் Rs 250 நிலுவையில் உள்ளது. எங்கள் முகவரிடம் UPI PIN சரிபார்க்கவும்",
    "Flipkart கேஷ்பேக் Rs 1500 தயார். UPI ID மற்றும் கார்டின் கடைசி 4 இலக்கங்கள் கொடுங்கள்",
    # Job / Loan scams
    "வீட்டில் இருந்து வேலை: Rs 15000 தினமும் சம்பாதிக்கலாம். Rs 500 பதிவு கட்டணம் UPI செலுத்தவும்",
    "Data entry வேலை: மணிக்கு Rs 800 வீட்டிலிருந்து. Rs 299 UPI மூலம் செலுத்தி தொடங்கவும்",
    "உடனடி கடன் Rs 5 லட்சம் அங்கீகரிக்கப்பட்டது. Rs 999 செயலாக்க கட்டணம் UPI மூலம் செலுத்தவும்",
    "Part time வேலை Rs 50000 மாதம். Rs 1000 பதிவு கட்டணம் UPI மூலம் செலுத்தவும் இப்போதே",
    # Courier scams
    "உங்கள் பார்சல் சுங்கத்தில் நிறுத்தப்பட்டுள்ளது. Rs 199 UPI மூலம் செலுத்தி விடுவிக்கவும்",
    "Amazon delivery நிலுவை: Rs 49 கையாளும் கட்டணம் UPI மூலம் செலுத்தி மீண்டும் திட்டமிடவும்",
    # SIM scams
    "உங்கள் SIM 24 மணி நேரத்தில் தடுக்கப்படும். கட்டாய மறு சரிபார்ப்புக்கு Aadhaar OTP பகிரவும்",
    "Airtel: உங்கள் எண் 9876543210 செயலிழக்கும். நிலுவை தொகை செலுத்த UPI PIN பகிரவும்",
    "Jio எச்சரிக்கை: SIM KYC காலாவதி. 8899776655 அழைத்து OTP பகிர்ந்து துண்டிக்கப்படுவதை தவிர்க்கவும்",
    # Misc
    "PhonePe: Rs 10000 போனஸ் உங்கள் கணக்கில் சேர்க்கப்பட்டது. பெற bit.ly/phonepe-bonus ல் OTP சரிபார்க்கவும்",
    "காப்பீட்டு கோரிக்கை Rs 75000 அங்கீகரிக்கப்பட்டது. ஸ்டாம்ப் டியூட்டி Rs 499 UPI மூலம் செலுத்தவும்",
    "உங்கள் மியூச்சுவல் ஃபண்ட் டிவிடெண்ட் Rs 3200 தயார். வங்கிக்கு மாற்ற UPI PIN பகிரவும்",
    "நீதிமன்ற அறிவிப்பு: UPI நிலுவை Rs 2300. உடனே செலுத்தவும் இல்லையேல் கணக்கு முடக்கப்படும்",
    "WINNER: PhonePe Rs 1 லட்ச பரிசு! Rs 199 செலுத்தி உங்கள் பரிசை கோரவும் 9876543210",
    "உங்கள் Paytm கணக்கு காலாவதியாகிறது. 1800-XXX-XXXX அழைத்து KYC புதுப்பிக்கவும் உடனே",
    "சந்தேகமான UPI உள்நுழைவு கண்டறியப்பட்டது. OTP பகிர்ந்து அங்கீகரிக்கப்படாத அணுகலை தடுக்கவும்",
    "RBI: UPI பரிவர்த்தனை வரம்பு அதிகரிக்கப்பட்டது. OTP பகிர்ந்து சரிபார்க்கவும் இப்போதே",
    "கிரிப்டோ முதலீடு: Rs 1000 ஐ 7 நாளில் Rs 50000 ஆக்குங்கள். UPI மூலம் அனுப்பவும்",
]

SCAM_MIXED = [
    "URGENT: உங்கள் account block ஆகும். Click here to update KYC now! Call 9876543210 immediately",
    "Congratulations! நீங்கள் Rs 50000 won! OTP அனுப்பவும் prize claim செய்ய இப்போதே",
    "PhonePe ALERT: உங்கள் KYC expired ஆகியது. Account suspended ஆகும். Update now bit.ly/kyc",
    "SBI: Account deactivate ஆகும் 24 hours-ல். OTP share பண்ணுங்க immediately reactivate பண்ண",
    "Lucky Winner! நீங்கள் iPhone வென்றீர்கள். Rs 199 shipping fee pay பண்ணவும் UPI மூலம்",
    "FRAUD ALERT: உங்கள் UPI-ல் suspicious transaction உள்ளது. OTP confirm பண்ணி cancel செய்யுங்க",
    "Income Tax refund Rs 15000 ready உள்ளது. UPI ID மற்றும் OTP கொடுங்கள் receive பண்ண",
    "Government scheme: Rs 5000 approved for you. UPI PIN கொடுங்கள் funds பெற இப்போதே",
    "Bank security: உங்கள் account compromised ஆகியது. PIN share பண்ணி secure பண்ணுங்க",
    "RBI notice: UPI re-verification required urgently. OTP பகிர்ந்து process complete பண்ணுங்க",
    "URGENT: உங்கள் Aadhaar KYC link expired. Update பண்ண bit.ly/aadhaar-link-ல் OTP enter பண்ணவும்",
    "Diwali offer: Rs 5000 cashback PhonePe-ல் ready. Claim பண்ண UPI PIN confirm பண்ணுங்கள் now",
    "Work from home job: Rs 800/hour earn பண்ணலாம். Registration fee Rs 299 UPI-ல் pay பண்ணவும்",
    "Gas cylinder subsidy Rs 300 approved. UPI PIN கொடுங்கள் refund receive பண்ண உடனே",
    "உங்கள் SIM card 24 hours-ல் block ஆகும். Aadhaar OTP share பண்ணி re-verify பண்ணுங்கள்",
    "HDFC Bank: Suspicious login detected. உங்கள் 6-digit PIN share பண்ணி account secure பண்ணுங்க",
    "Amazon delivery: Parcel held at customs. Rs 199 pay பண்ண UPI link: bit.ly/amz-custom",
    "PM Kisan: Rs 6000 government funds ready. UPI PIN verify பண்ணி funds receive பண்ணுங்கள்",
    "Flipkart cashback Rs 1500 ready. UPI ID மற்றும் card last 4 digits கொடுங்கள் receive பண்ண",
    "WINNER: நீங்கள் Rs 10 lakh prize வென்றீர்கள்! OTP கொடுங்கள் prize transfer பண்ண இப்போதே",
    "Court notice: UPI outstanding dues Rs 2300. Immediately pay பண்ணவும் account seized ஆகாமல்",
    "Instant loan Rs 2 lakh approved. No documents. Processing fee Rs 500 UPI-ல் pay பண்ணவும்",
    "உங்கள் mutual fund dividend Rs 3200 ready. Transfer பண்ண UPI PIN share பண்ணுங்கள் now",
    "Crypto invest: Rs 1000 to Rs 50000 in 7 days. UPI-ல் அனுப்பவும் இப்போதே profit பெற",
    "EPFO: PF withdrawal Rs 50000 ready. UPI PIN confirm பண்ணி bank-ல் transfer பண்ணுங்கள்",
    "Election bonus Rs 500 for Voter ID linking. UPI PIN share பண்ணி receive பண்ணுங்கள் today",
    "Insurance claim Rs 75000 approved. Stamp duty Rs 499 UPI-ல் pay பண்ணவும் cheque release பண்ண",
    "Part time job: Rs 50000/month. Registration Rs 1000 UPI-ல் pay பண்ணவும் இப்போதே apply பண்ண",
    "Jio SIM KYC expired. 8899776655 call பண்ணி OTP share பண்ணி disconnection தவிர்க்கவும்",
    "PM Awas Yojana: Rs 2.5 lakh housing subsidy approved. UPI PIN கொடுங்கள் funds release பண்ண",
]

# ── Legitimate messages ───────────────────────────────────────────────────────

LEGIT_ENGLISH = [
    # Bank credit / debit
    "Rs 5000.00 credited to your account XXXXXXXX1234 via UPI on 03-Apr-2026. Ref: UPI123456789",
    "Your UPI payment of Rs 200 to merchant BigBazaar was successful. Transaction ID: TXN789012345",
    "HDFC Bank: UPI payment of Rs 10000 to XYZ Trading Co. successful. Avail. bal: Rs 25432.50",
    "ICICI Bank: ECS debit of Rs 1500 for loan EMI processed. Outstanding balance: Rs 45000.",
    "Canara Bank: Salary credit of Rs 45000 to your account on 01-Apr-2026. A/c: XXXXXXXX.",
    "NEFT transfer of Rs 25000 to HDFC account successful. UTR: NEFT20260403XXXXXX.",
    "ATM cash withdrawal of Rs 3000 from your account on 03-Apr at SBI ATM Chennai T. Nagar.",
    "Your SIP of Rs 2000 has been processed successfully. Units credited to your mutual fund.",
    "LIC premium of Rs 5000 for policy XXXXXXX received. Next due date: 01-Jul-2026.",
    "Your UPI transaction of Rs 100 to PM-CARES Fund was successful. Thank you for contributing.",
    "School fees Rs 8500 paid via UPI for Arjun Sharma (Class 10). Receipt no: SCH20260403.",
    "CRED: Rs 5000 bill payment successful for your HDFC credit card. Points earned: 500",
    "Groww: Rs 500 invested in Nifty 50 Index Fund. NAV: Rs 125.50. Units allotted: 3.98.",
    "Reliance Digital: Rs 12000 paid via UPI for Samsung TV. Order confirmed. Delivery in 3 days.",
    "Your Aadhaar-enabled payment of Rs 500 at PDS outlet was successful. Thank you for using AEPS.",
    "IndiGo: Ticket booked. Rs 4500 paid via UPI. PNR: XXXXXX. Departure: 10-Apr-2026 at 0600 hrs.",
    "Axis Bank: Rs 3000 transferred to savings account via UPI. Available balance: Rs 18500.",
    "Kotak Bank: Auto-debit of Rs 2500 for credit card bill on 03-Apr-2026. Thank you.",
    "PNB: Salary credit Rs 38000 on 01-Apr-2026. Balance: Rs 42500. A/c: XXXXXXXX1234.",
    "Your FD of Rs 50000 matured. Amount credited to your account. Ref: FD20260403XXXX.",
    # OTP messages (legitimate)
    "OTP for your UPI transaction is 845621. Valid for 10 minutes. Do NOT share with anyone.",
    "OTP is 234567 for your SBI netbanking login. This OTP is valid for 5 minutes only.",
    "Your OTP for Paytm transaction is 112233. Valid for 3 minutes. Never share OTP with anyone.",
    "HDFC: OTP for your credit card transaction is 778899. Valid 10 mins. Do not share with anyone.",
    "PhonePe OTP: 334455. Use this to verify your payment. Valid 5 minutes. Don't share with anyone.",
    # Food / delivery
    "PhonePe: Rs 1500 sent to Rahul Kumar (ra***l@ybl). Transaction ref: PTM20260403112233",
    "Amazon Pay: Order #12345 payment of Rs 899 received successfully via UPI. Thank you!",
    "Swiggy payment of Rs 350 successful. Enjoy your meal! Order will arrive in approximately 30 min.",
    "Your Google Pay balance: Rs 1250. Last transaction: Rs 50 to Metro Station on 03-Apr-2026.",
    "Zomato: Your order is confirmed. Rs 420 paid via UPI. Estimated delivery in 45 minutes.",
    "BigBasket order Rs 1200 placed successfully. Payment via UPI confirmed. Delivery tomorrow.",
    "Zepto: Rs 750 grocery order confirmed. Payment via UPI. Delivery in approximately 10 minutes.",
    "Myntra: Order #98765 Rs 1499 payment successful via UPI. Delivery by 06-Apr-2026.",
    # Travel
    "Uber: Trip payment of Rs 180 charged to your UPI. Receipt has been sent to your email.",
    "Ola: Ride completed. Rs 95 charged via UPI. Please rate your driver to help us improve.",
    "MakeMyTrip: Booking confirmed. Rs 8500 paid via UPI for flight DEL-BOM on 10-Apr-2026.",
    "Paytm: Rs 500 auto-debited for your electricity bill. Receipt available in the Paytm app.",
    "BookMyShow: 2 tickets for Movie confirmed. Rs 600 paid via UPI. Enjoy the show!",
    "IRCTC: Ticket booked PNR XXXXXXXX. Rs 1200 paid via UPI. Train 12345 on 10-Apr-2026.",
    # Telecom
    "Airtel: Rs 399 recharge successful for 9876543210. Validity: 84 days. Data: 1.5GB/day.",
    "Jio: Monthly pack Rs 299 renewed for 9988776655. Validity extended by 28 days.",
    "Your PhonePe transaction of Rs 1000 to BSNL Postpaid bill was successful on 03-Apr-2026.",
    "Vi: Rs 249 recharge successful. Validity 28 days. Data: 1GB/day. Balance: Rs 0.",
    # Finance / investments
    "PhonePe: Split bill payment of Rs 250 from Priya received. Total collected: Rs 750.",
    "Zerodha: Rs 5000 added to your trading account via UPI. Available margin updated.",
    "Groww: SIP of Rs 1000 processed for HDFC Mid-Cap Fund on 03-Apr-2026. Units allotted: 8.2",
    "HDFC Life: Premium of Rs 8000 received for policy no. XXXXXXXX. Next due: Oct 2026.",
    "NPS: Contribution of Rs 2000 credited to your Tier-1 account. PRAN: XXXXXXXXXXXX.",
    "SBI Cards: Payment of Rs 12000 received for card ending 5678. Outstanding: Rs 0.",
    "Your PPF deposit of Rs 5000 for FY 2026-27 is successful. Balance: Rs 1,25,000.",
    # Misc
    "Your UPI ID ra***a@okaxis is now active. Use it to send and receive money instantly.",
    "HDFC: Standing instruction of Rs 5000 to RD account executed. Balance updated.",
    "GPay: Rs 2000 received from Suresh Kumar. Your new balance: Rs 4500.",
    "Paytm Postpaid: Bill of Rs 3200 due on 15-Apr. Pay via UPI to avoid late fee.",
    "Amazon: Rs 299 refund for cancelled order #567890 credited to your UPI. Ref: REF20260403",
]

LEGIT_TAMIL = [
    # Bank credit / debit
    "Rs 5000.00 உங்கள் கணக்கில் XXXXXXXX1234 UPI மூலம் வரவு வைக்கப்பட்டது. Ref: UPI123456",
    "UPI மூலம் Rs 200 BigBazaar-க்கு வெற்றிகரமாக அனுப்பப்பட்டது. பரிமாற்ற எண்: TXN789012",
    "HDFC வங்கி: Rs 10000 XYZ நிறுவனத்திற்கு UPI வழியாக வெற்றிகரமாக அனுப்பப்பட்டது.",
    "உங்கள் சம்பளம் Rs 45000 01-ஏப்ரல்-2026 அன்று கணக்கில் வரவு வைக்கப்பட்டது.",
    "ATM ரொக்க திரும்பல் Rs 3000 SBI ATM சென்னை ஆம்பத்தூரில் 03-ஏப்ரல் அன்று.",
    "NEFT பரிமாற்றம் Rs 25000 HDFC கணக்கிற்கு வெற்றிகரமாக. UTR: NEFT20260403XXXXXX.",
    "Axis வங்கி: Rs 3000 UPI வழியாக சேமிப்புக் கணக்கிற்கு மாற்றப்பட்டது. இருப்பு: Rs 18500.",
    "Kotak வங்கி: கிரெடிட் கார்டு கட்டணம் Rs 2500 03-ஏப்ரல்-2026 அன்று தானாக கழிக்கப்பட்டது.",
    "PNB: சம்பளம் Rs 38000 வரவு வைக்கப்பட்டது 01-ஏப்ரல்-2026. இருப்பு: Rs 42500.",
    "உங்கள் FD Rs 50000 முதிர்ந்தது. தொகை கணக்கில் வரவு வைக்கப்பட்டது. Ref: FD20260403",
    "ICICI வங்கி: கடன் EMI Rs 1500 ECS மூலம் கழிக்கப்பட்டது. நிலுவை: Rs 45000.",
    "Canara வங்கி: சம்பளம் Rs 38000 01-ஏப்ரல்-2026 அன்று உங்கள் கணக்கில் வரவு வைக்கப்பட்டது.",
    # OTP messages (legitimate)
    "உங்கள் OTP: 845621. 10 நிமிடங்களுக்கு செல்லுபடியாகும். யாருடனும் பகிர வேண்டாம்.",
    "SBI NetBanking OTP: 234567. இது 5 நிமிடங்களுக்கு மட்டும் செல்லுபடியாகும்.",
    "Paytm பரிமாற்றத்திற்கான OTP: 112233. 3 நிமிடங்களுக்கு செல்லுபடியாகும். யாரிடமும் பகிர வேண்டாம்.",
    "HDFC: கிரெடிட் கார்டு பரிமாற்றத்திற்கான OTP 778899. 10 நிமிடம். யாரிடமும் பகிர வேண்டாம்.",
    "PhonePe OTP: 334455. பணம் செலுத்த இதை பயன்படுத்தவும். 5 நிமிடம் செல்லுபடியாகும்.",
    # Food / delivery
    "PhonePe: Rs 1500 ராஹுல் குமாருக்கு வெற்றிகரமாக அனுப்பப்பட்டது. Ref: PTM20260403",
    "Amazon Pay: ஆர்டர் #12345 Rs 899 UPI மூலம் வெற்றிகரமாக பெறப்பட்டது. நன்றி!",
    "Swiggy பணம் Rs 350 வெற்றிகரமாக செலுத்தப்பட்டது. சாப்பாடு 30 நிமிடத்தில் வரும்.",
    "Zomato: ஆர்டர் உறுதிப்படுத்தப்பட்டது. Rs 420 UPI மூலம் செலுத்தப்பட்டது. 45 நிமிடம்.",
    "BigBasket Rs 1200 ஆர்டர் உறுதி. UPI பணம் உறுதிப்படுத்தப்பட்டது. நாளை டெலிவரி.",
    "Zepto: Rs 750 மளிகை ஆர்டர் உறுதிப்படுத்தப்பட்டது. UPI பணம் கழிக்கப்பட்டது.",
    "Myntra ஆர்டர் #98765 Rs 1499 UPI மூலம் வெற்றிகரமாக செலுத்தப்பட்டது.",
    # Travel
    "Uber: பயண கட்டணம் Rs 180 உங்கள் UPI மூலம் வசூலிக்கப்பட்டது.",
    "Ola: பயணம் முடிந்தது. Rs 95 UPI மூலம் வசூலிக்கப்பட்டது. மதிப்பீடு செய்யவும்.",
    "IndiGo: டிக்கெட் பதிவு செய்யப்பட்டது. Rs 4500 UPI மூலம் செலுத்தப்பட்டது. PNR: XXXXXX.",
    "Paytm: மின்சாரக் கட்டணம் Rs 500 தானாகவே கட்டப்பட்டது. ரசீது Paytm-ல் உள்ளது.",
    "BookMyShow: 2 டிக்கெட்டுகள் உறுதிப்படுத்தப்பட்டன. Rs 600 UPI மூலம் செலுத்தப்பட்டது.",
    "IRCTC: ரயில் டிக்கெட் PNR XXXXXXXX. Rs 1200 UPI மூலம் செலுத்தப்பட்டது.",
    # Telecom
    "Airtel: Rs 399 ரீசார்ஜ் வெற்றிகரமாக 9876543210-க்கு. செல்லுபடி: 84 நாட்கள்.",
    "Jio: மாத பேக் Rs 299 9988776655-க்கு புதுப்பிக்கப்பட்டது. 28 நாட்கள் நீட்டிக்கப்பட்டது.",
    "Vi: Rs 249 ரீசார்ஜ் வெற்றிகரம். செல்லுபடி 28 நாட்கள். Data: 1GB/day.",
    # Finance / investments
    "LIC பிரீமியம் Rs 5000 நோக்கம் XXXXXXX-க்காக பெறப்பட்டது. அடுத்த தேதி: 01-Jul-2026.",
    "Groww: Nifty 50 Index Fund-ல் Rs 500 முதலீடு செய்யப்பட்டது. NAV: Rs 125.50.",
    "PhonePe: பிரிந்து கட்டணம் Rs 250 பிரியாவிடமிருந்து பெறப்பட்டது. மொத்தம்: Rs 750.",
    "PM-CARES நிதிக்கு Rs 100 UPI பரிமாற்றம் வெற்றிகரமாக முடிந்தது. நன்றி.",
    "பள்ளிக் கட்டணம் Rs 8500 UPI மூலம் அர்ஜுன் சர்மாவிற்காக (வகுப்பு 10) செலுத்தப்பட்டது.",
    "NPS: Rs 2000 Tier-1 கணக்கில் வரவு வைக்கப்பட்டது. PRAN: XXXXXXXXXXXX.",
    "SBI Cards: Rs 12000 கார்டு 5678 கட்டணம் பெறப்பட்டது. நிலுவை: Rs 0.",
    "உங்கள் PPF வைப்பு Rs 5000 FY 2026-27-க்கு வெற்றிகரம். இருப்பு: Rs 1,25,000.",
    "Zerodha: Rs 5000 UPI மூலம் வர்த்தக கணக்கில் சேர்க்கப்பட்டது. மார்ஜின் புதுப்பிக்கப்பட்டது.",
    "HDFC Life: பாலிசி XXXXXXXX-க்கு பிரீமியம் Rs 8000 பெறப்பட்டது. அடுத்தது: Oct 2026.",
    "Google Pay: Rs 2000 சுரேஷ் குமாரிடமிருந்து பெறப்பட்டது. புதிய இருப்பு: Rs 4500.",
    # Misc
    "Amazon: ரத்து செய்யப்பட்ட ஆர்டர் #567890-க்கு Rs 299 திரும்பப் பெறல் UPI-ல் வரவு வைக்கப்பட்டது.",
    "GPay: Rs 500 Google Pay balance-ல் add ஆகியது. Total: Rs 1250. 03-Apr-2026.",
    "உங்கள் UPI ID ra***a@okaxis இப்போது செயலில் உள்ளது. பணம் அனுப்ப மற்றும் பெற பயன்படுத்தவும்.",
    "CRED: HDFC கிரெடிட் கார்டு கட்டணம் Rs 5000 வெற்றிகரம். Points earned: 500.",
    "Groww: HDFC Mid-Cap Fund SIP Rs 1000 03-ஏப்ரல்-2026 அன்று. Units allotted: 8.2.",
]

LEGIT_MIXED = [
    "Payment successful! Rs 500 உங்கள் கணக்கில் credited ஆகியது. Ref: TXN123456789",
    "Your order confirmed. Rs 350 Swiggy-க்கு UPI மூலம் வெற்றிகரமாக செலுத்தப்பட்டது.",
    "OTP for transaction is 567890. 10 நிமிடங்களுக்கு valid. யாரிடமும் share பண்ணாதீர்கள்.",
    "HDFC: Rs 2000 EMI deducted ஆகியது இன்று. Available balance: Rs 15000.",
    "Uber trip Rs 120 UPI மூலம் paid ஆகியது. Driver-ஐ rate பண்ணுங்கள் தயவுசெய்து.",
    "Amazon order #67890 Rs 1299 payment confirmed. Delivery நாளை வரும்.",
    "Airtel recharge Rs 239 successful for your number. Validity: 28 நாட்கள்.",
    "Google Pay: Rs 800 to Priya Sharma UPI மூலம் successful ஆகியது. Transaction complete.",
    "Paytm: Rs 350 electricity bill paid successfully. Receipt available in the app உள்ளது.",
    "BigBasket Rs 950 order confirmed via UPI. Tomorrow delivery வரும் தயவுசெய்து காத்திருங்கள்.",
    "Salary Rs 42000 உங்கள் account-ல் credited ஆகியது 01-Apr-2026. SBI Bank.",
    "Zepto Rs 680 grocery order UPI-ல் paid ஆகியது. 10 mins-ல் delivery வரும்.",
    "IRCTC ticket booked. Rs 980 UPI-ல் paid. PNR: XXXXXXXX. Journey: 10-Apr-2026.",
    "Groww: Rs 1000 SIP Nifty 50-ல் processed ஆகியது. Units credited to your account.",
    "Ola ride completed. Rs 85 UPI-ல் charged. Thank you for riding! Rate your experience.",
    "IndiGo flight confirmed. Rs 3500 UPI-ல் paid. PNR: XXXXXX. Departure 06:00 hrs.",
    "CRED: Credit card bill Rs 8000 paid successfully. Points earned: 800.",
    "PhonePe: Rs 3000 Suresh-ஐ send பண்ணினீர்கள். Transaction ID: PPE20260403XXXX.",
    "LIC premium Rs 6000 received for policy XXXXXXX. Next due: July 2026. நன்றி.",
    "NPS contribution Rs 2000 Tier-1 account-ல் credited ஆகியது. PRAN: XXXXXXXXXXXX.",
    "SBI: FD Rs 1 lakh matured. Amount credited to your account. Congrats! Ref: FD2026XXX.",
    "Amazon refund Rs 499 for order #345678 cancelled. UPI-ல் credited ஆகியது. Ref: REF2026",
    "Jio recharge Rs 349 successful. Data: 2GB/day. Validity: 56 நாட்கள். Thank you.",
    "Vi postpaid bill Rs 599 paid via UPI. Due amount: Rs 0. Thank you for paying on time.",
    "BookMyShow: 3 tickets confirmed. Rs 900 UPI-ல் paid. Enjoy the movie! Ref: BMS20260403",
    "MakeMyTrip: Hotel booking confirmed. Rs 4200 UPI-ல் paid. Check-in: 10-Apr-2026.",
    "Zerodha: Rs 10000 trading account-ல் added via UPI. Available margin updated.",
    "HDFC Life insurance premium Rs 5000 received. Policy XXXXXXXX. Next: Oct 2026.",
    "Kotak: Standing instruction Rs 3000 RD account-க்கு executed. Balance updated.",
    "GPay: Rs 1500 received from Kavitha. உங்கள் balance: Rs 3200. 03-Apr-2026.",
]

SCAM_HINDI = [
    # KYC / Account block threats
    "तुरंत: आपका UPI खाता ब्लॉक हो जाएगा! KYC अपडेट करें। अभी कॉल करें 9876543210",
    "आपका Paytm KYC समाप्त हो गया। खाता 2 घंटे में निलंबित होगा। अभी अपडेट करें",
    "Google Pay KYC अधूरा है। कल खाता ब्लॉक होगा। तुरंत अपडेट करें",
    "NPCI चेतावनी: आपकी UPI ID निष्क्रिय हो जाएगी। 24 घंटे में अपडेट करें",
    "आपका SBI खाता अधूरे KYC के कारण बंद होगा। पुनः सक्रिय करने के लिए विवरण साझा करें",
    "आधार लिंक UPI KYC समाप्त। 9988776655 पर कॉल करें और पुनः सक्रिय करें",
    "HDFC बैंक: UPI KYC सत्यापन लंबित है। bit.ly/hdfc-kyc पर अपडेट करें",
    "अत्यावश्यक: 24 घंटे में KYC अपडेट न करने पर PhonePe खाता निलंबित होगा",
    "SBI: KYC सत्यापन लंबित। खाता प्रतिबंधित। bit.ly/sbi-kyc पर जाएं या 8899001122 पर कॉल करें",
    # OTP / PIN requests
    "SBI चेतावनी: 24 घंटे में खाता निष्क्रिय होगा। पुनः सक्रिय करने के लिए OTP साझा करें",
    "सरकारी योजना: Rs 5000 सहायता स्वीकृत। धन प्राप्त करने के लिए UPI PIN दें",
    "HDFC बैंक: संदिग्ध गतिविधि। खाता सुरक्षित करने के लिए अभी UPI PIN साझा करें",
    "मुफ्त ऑफर: Rs 2000 PhonePe वॉलेट में। सक्रिय करने के लिए UPI PIN साझा करें",
    "आपका खाता खतरे में है। धन सुरक्षित करने के लिए तुरंत OTP साझा करें",
    "ICICI बैंक: अधूरा सत्यापन। खाता अनलॉक करने के लिए अपना 6 अंकों का PIN दें",
    "बैंक सुरक्षा टीम: संदिग्ध UPI अनुरोध। इसे रोकने के लिए PIN साझा करें अभी",
    # Prize / lottery
    "बधाई हो! आपने Rs 50000 का पुरस्कार जीता। दावा करने के लिए OTP 5534 भेजें",
    "लॉटरी विजेता! Rs 25 लाख जीते। पुरस्कार पाने के लिए Rs 500 जमा करें अभी",
    "Diwali ऑफर: UPI पर Rs 5000 कैशबैक जीते। पाने के लिए OTP 7652 भेजें",
    "PhonePe वार्षिकोत्सव लॉटरी विजेता के रूप में आप चुने गए। Rs 1 लाख के लिए OTP भेजें",
    "Amazon ऑफर: Rs 5000 वाउचर जीते। खाते में पाने के लिए बैंक OTP साझा करें",
    # Government / fake schemes
    "PM किसान योजना: Rs 6000 तैयार। सरकारी धन पाने के लिए UPI PIN दें",
    "आयकर विभाग: Rs 15000 कर वापसी लंबित। तत्काल क्रेडिट के लिए UPI ID और OTP दें",
    "COVID राहत कोष Rs 3500 स्वीकृत। UPI ID और OTP साझा करें",
    "EPFO: PF निकासी Rs 50000 तैयार। बैंक में ट्रांसफर के लिए UPI PIN दें",
    "PM Awas Yojana: Rs 2.5 लाख आवास सब्सिडी स्वीकृत। धन जारी करने के लिए UPI PIN भेजें",
    # Refund scams
    "बिजली बिल वापसी Rs 2000 UPI के माध्यम से। प्राप्त करने के लिए PIN दर्ज करें",
    "गैस सिलिंडर सब्सिडी Rs 300 सरकार ने मंजूर की। UPI PIN दें प्राप्त करने के लिए",
    "LPG सब्सिडी Rs 250 लंबित। हमारे एजेंट के साथ UPI PIN सत्यापित करें",
    "Flipkart कैशबैक Rs 1500 तैयार। UPI ID और कार्ड के अंतिम 4 अंक दें",
]

LEGIT_HINDI = [
    # Bank credit / debit
    "Rs 5000.00 आपके खाते XXXXXXXX1234 में UPI द्वारा जमा किया गया। Ref: UPI123456",
    "UPI द्वारा Rs 200 BigBazaar को सफलतापूर्वक भेजा गया। लेन-देन संख्या: TXN789012",
    "HDFC बैंक: Rs 10000 XYZ कंपनी को UPI के माध्यम से सफलतापूर्वक भेजा गया।",
    "आपका वेतन Rs 45000 01-अप्रैल-2026 को खाते में जमा किया गया।",
    "ATM नकद निकासी Rs 3000 SBI ATM दिल्ली में 03-अप्रैल को।",
    "Axis बैंक: Rs 3000 UPI के माध्यम से बचत खाते में स्थानांतरित। शेष: Rs 18500.",
    "PNB: वेतन Rs 38000 जमा 01-अप्रैल-2026. शेष: Rs 42500.",
    "ICICI बैंक: लोन EMI Rs 1500 ECS द्वारा काटी गई। शेष: Rs 45000.",
    "Canara बैंक: वेतन Rs 38000 01-अप्रैल-2026 को आपके खाते में जमा।",
    # OTP messages (legitimate)
    "आपका OTP: 845621. 10 मिनट के लिए वैध। किसी के साथ साझा न करें।",
    "SBI NetBanking OTP: 234567. यह केवल 5 मिनट के लिए वैध है।",
    "Paytm लेन-देन के लिए OTP: 112233. 3 मिनट के लिए वैध। किसी को न बताएं।",
    "HDFC: क्रेडिट कार्ड लेन-देन के लिए OTP 778899. 10 मिनट। किसी को न बताएं।",
    # Food / delivery / travel
    "PhonePe: Rs 1500 राहुल कुमार को सफलतापूर्वक भेजा गया। Ref: PTM20260403",
    "Amazon Pay: ऑर्डर #12345 Rs 899 UPI द्वारा सफलतापूर्वक प्राप्त। धन्यवाद!",
    "Swiggy भुगतान Rs 350 सफलतापूर्वक किया गया। खाना 30 मिनट में आएगा।",
    "Uber: यात्रा शुल्क Rs 180 आपके UPI से लिया गया।",
    "IndiGo: टिकट बुक। Rs 4500 UPI से भुगतान। PNR: XXXXXX.",
    "IRCTC: रेल टिकट PNR XXXXXXXX. Rs 1200 UPI से भुगतान।",
    "BookMyShow: 2 टिकट कन्फर्म। Rs 600 UPI से भुगतान। शो का आनंद लें!",
    # Telecom / Finance
    "Airtel: Rs 399 रिचार्ज सफल 9876543210 के लिए। वैधता: 84 दिन।",
    "Jio: मासिक पैक Rs 299 9988776655 के लिए नवीनीकृत। 28 दिन बढ़ाए गए।",
    "Groww: Nifty 50 Index Fund में Rs 500 निवेश। NAV: Rs 125.50.",
    "LIC प्रीमियम Rs 5000 पॉलिसी XXXXXXX के लिए प्राप्त। अगली तारीख: 01-Jul-2026.",
    "SBI Cards: कार्ड 5678 के लिए Rs 12000 भुगतान प्राप्त। शेष: Rs 0.",
    "Zerodha: Rs 5000 UPI द्वारा ट्रेडिंग खाते में जमा। मार्जिन अपडेट।",
    "Amazon: रद्द ऑर्डर #567890 के लिए Rs 299 रिफंड UPI में जमा। Ref: REF2026",
]

SCAM_TELUGU = [
    # KYC / Account block threats
    "అత్యవసరం: మీ UPI ఖాతా బ్లాక్ అవుతుంది! KYC అప్‌డేట్ చేయండి. ఇప్పుడే కాల్ చేయండి 9876543210",
    "మీ Paytm KYC గడువు ముగిసింది. ఖాతా నిలిపివేయబడుతుంది. ఇప్పుడే అప్‌డేట్ చేయండి",
    "Google Pay KYC పూర్తి కాలేదు. రేపు ఖాతా బ్లాక్ అవుతుంది. వెంటనే అప్‌డేట్ చేయండి",
    "NPCI హెచ్చరిక: మీ UPI ID నిలిపివేయబడుతుంది. 24 గంటల్లో అప్‌డేట్ చేయండి",
    "మీ SBI ఖాతా అసంపూర్ణ KYC కారణంగా మూసివేయబడుతుంది. వివరాలు పంచుకోండి",
    "ఆధార్ లింక్ చేసిన UPI KYC గడువు ముగిసింది. 9988776655 కాల్ చేసి మళ్ళీ యాక్టివేట్ చేయండి",
    "HDFC బ్యాంక్: మీ UPI KYC వెరిఫికేషన్ పెండింగ్‌లో ఉంది. bit.ly/hdfc-kyc లో అప్‌డేట్ చేయండి",
    "అత్యవసరం: 24 గంటల్లో KYC అప్‌డేట్ చేయకపోతే మీ PhonePe ఖాతా నిలిపివేయబడుతుంది",
    # OTP / PIN requests
    "SBI హెచ్చరిక: 24 గంటల్లో ఖాతా నిలిపివేయబడుతుంది. OTP పంచుకోండి మళ్ళీ యాక్టివేట్ చేయడానికి",
    "ప్రభుత్వ పథకం: Rs 5000 సహాయం మంజూరైంది. UPI PIN ఇవ్వండి డబ్బు పొందడానికి",
    "HDFC బ్యాంక్: అనుమానాస్పద కార్యకలాపం. UPI PIN పంచుకోండి ఖాతాను సురక్షితం చేయడానికి",
    "ఉచిత ఆఫర్: Rs 2000 PhonePe వాలెట్‌లో. UPI PIN పంచుకోండి యాక్టివేట్ చేయడానికి",
    "మీ ఖాతా రాజీ పడింది. OTP పంచుకోండి డబ్బు సురక్షితం చేయడానికి వెంటనే",
    "ICICI బ్యాంక్: అసంపూర్ణ వెరిఫికేషన్. మీ 6 అంకెల PIN ఇవ్వండి ఖాతా అన్‌లాక్ చేయడానికి",
    # Prize / lottery
    "అభినందనలు! మీరు Rs 50000 బహుమతి గెలిచారు. OTP పంపండి 5534 క్లెయిమ్ చేయడానికి",
    "లాటరీ విజేత! Rs 25 లక్షలు గెలిచారు. Rs 500 చెల్లించి బహుమతి పొందండి ఇప్పుడే",
    "Diwali ఆఫర్: UPI ద్వారా Rs 5000 క్యాష్‌బ్యాక్ గెలిచారు. OTP 7652 పంపండి పొందడానికి",
    "PhonePe వార్షికోత్సవ లాటరీ విజేతగా మీరు ఎంపికయ్యారు. Rs 1 లక్ష పొందడానికి OTP పంపండి",
    "అమెజాన్ ఆఫర్: Rs 5000 వోచర్ గెలిచారు. బ్యాంక్ OTP పంచుకోండి ఖాతాలో పొందడానికి",
    # Government / fake schemes
    "PM కిసాన్ పథకం: Rs 6000 సిద్ధంగా ఉంది. ప్రభుత్వ డబ్బు పొందడానికి UPI PIN ఇవ్వండి",
    "ఆదాయపు పన్ను విభాగం: Rs 12000 రీఫండ్ పెండింగ్. UPI ID మరియు OTP ఇవ్వండి",
    "COVID నివారణ నిధి Rs 3500 మంజూరైంది. UPI ID మరియు OTP ఇవ్వండి",
    "EPFO: PF విత్‌డ్రాయల్ Rs 50000 సిద్ధంగా ఉంది. బ్యాంక్‌కు బదిలీ చేయడానికి UPI PIN ఇవ్వండి",
    # Refund scams
    "విద్యుత్ బిల్లు రీఫండ్ Rs 2000 UPI ద్వారా. పొందడానికి PIN నమోదు చేయండి",
    "గ్యాస్ సిలిండర్ సబ్సిడీ Rs 300 ప్రభుత్వం మంజూరు చేసింది. UPI PIN ఇవ్వండి పొందడానికి",
    "LPG సబ్సిడీ Rs 250 పెండింగ్‌లో ఉంది. మా ఏజెంట్‌తో UPI PIN వెరిఫై చేయండి",
    "Flipkart క్యాష్‌బ్యాక్ Rs 1500 సిద్ధంగా ఉంది. UPI ID మరియు కార్డ్ చివరి 4 అంకెలు ఇవ్వండి",
]

LEGIT_TELUGU = [
    # Bank credit / debit
    "Rs 5000.00 మీ ఖాతా XXXXXXXX1234 లో UPI ద్వారా జమ చేయబడింది. Ref: UPI123456",
    "UPI ద్వారా Rs 200 BigBazaar కు విజయవంతంగా పంపబడింది. లావాదేవీ సంఖ్య: TXN789012",
    "HDFC బ్యాంక్: Rs 10000 XYZ కంపెనీకి UPI ద్వారా విజయవంతంగా పంపబడింది.",
    "మీ జీతం Rs 45000 01-ఏప్రిల్-2026 న ఖాతాలో జమ చేయబడింది.",
    "ATM నగదు విత్‌డ్రాయల్ Rs 3000 SBI ATM హైదరాబాద్‌లో 03-ఏప్రిల్ న.",
    "Axis బ్యాంక్: Rs 3000 UPI ద్వారా సేవింగ్స్ ఖాతాకు బదిలీ చేయబడింది. బాలెన్స్: Rs 18500.",
    "PNB: జీతం Rs 38000 జమ చేయబడింది 01-ఏప్రిల్-2026. బాలెన్స్: Rs 42500.",
    "ICICI బ్యాంక్: లోన్ EMI Rs 1500 ECS ద్వారా కట్ చేయబడింది. మిగిలినది: Rs 45000.",
    # OTP messages (legitimate)
    "మీ OTP: 845621. 10 నిమిషాలు చెల్లుబాటు అవుతుంది. ఎవరితోనూ పంచుకోవద్దు.",
    "SBI NetBanking OTP: 234567. ఇది 5 నిమిషాలు మాత్రమే చెల్లుబాటు అవుతుంది.",
    "Paytm లావాదేవీకి OTP: 112233. 3 నిమిషాలు చెల్లుబాటు. ఎవరికీ చెప్పవద్దు.",
    "HDFC: క్రెడిట్ కార్డ్ లావాదేవీకి OTP 778899. 10 నిమిషాలు. ఎవరికీ చెప్పవద్దు.",
    # Food / delivery / travel
    "PhonePe: Rs 1500 రాహుల్ కుమార్‌కు విజయవంతంగా పంపబడింది. Ref: PTM20260403",
    "Amazon Pay: ఆర్డర్ #12345 Rs 899 UPI ద్వారా విజయవంతంగా స్వీకరించబడింది. ధన్యవాదాలు!",
    "Swiggy పేమెంట్ Rs 350 విజయవంతంగా చెల్లించబడింది. ఆహారం 30 నిమిషాల్లో వస్తుంది.",
    "Zomato: ఆర్డర్ నిర్ధారించబడింది. Rs 420 UPI ద్వారా చెల్లించబడింది. 45 నిమిషాలు.",
    "Uber: ప్రయాణ చార్జ్ Rs 180 మీ UPI ద్వారా వసూలు చేయబడింది.",
    "IndiGo: టికెట్ బుక్ చేయబడింది. Rs 4500 UPI ద్వారా చెల్లించబడింది. PNR: XXXXXX.",
    "IRCTC: రైలు టికెట్ PNR XXXXXXXX. Rs 1200 UPI ద్వారా చెల్లించబడింది.",
    # Telecom / Finance
    "Airtel: Rs 399 రీచార్జ్ విజయవంతంగా 9876543210 కు. చెల్లుబాటు: 84 రోజులు.",
    "Jio: మంత్లీ పేక్ Rs 299 9988776655 కు రెన్యూ చేయబడింది. 28 రోజులు పొడిగించబడింది.",
    "Groww: Nifty 50 Index Fund లో Rs 500 పెట్టుబడి పెట్టబడింది. NAV: Rs 125.50.",
    "LIC ప్రీమియం Rs 5000 పాలసీ XXXXXXX కోసం స్వీకరించబడింది. తదుపరి తేదీ: 01-Jul-2026.",
    "SBI Cards: కార్డ్ 5678 కోసం Rs 12000 పేమెంట్ స్వీకరించబడింది. మిగిలినది: Rs 0.",
    "Zerodha: Rs 5000 UPI ద్వారా ట్రేడింగ్ ఖాతాలో జమ చేయబడింది. మార్జిన్ అప్‌డేట్ చేయబడింది.",
    "Amazon: రద్దు చేసిన ఆర్డర్ #567890 కోసం Rs 299 రీఫండ్ UPI లో జమ చేయబడింది.",
]


def generate():
    # ── 1. Synthetic UPI-specific messages ────────────────────────────────────
    rows = []
    for t in SCAM_ENGLISH:
        rows.append({"text": t, "label": 1, "language": "English"})
    for t in SCAM_HINDI:
        rows.append({"text": t, "label": 1, "language": "Hindi"})
    for t in SCAM_TAMIL:
        rows.append({"text": t, "label": 1, "language": "Tamil"})
    for t in SCAM_TELUGU:
        rows.append({"text": t, "label": 1, "language": "Telugu"})
    for t in SCAM_MIXED:
        rows.append({"text": t, "label": 1, "language": "Mixed"})
    for t in LEGIT_ENGLISH:
        rows.append({"text": t, "label": 0, "language": "English"})
    for t in LEGIT_HINDI:
        rows.append({"text": t, "label": 0, "language": "Hindi"})
    for t in LEGIT_TAMIL:
        rows.append({"text": t, "label": 0, "language": "Tamil"})
    for t in LEGIT_TELUGU:
        rows.append({"text": t, "label": 0, "language": "Telugu"})
    for t in LEGIT_MIXED:
        rows.append({"text": t, "label": 0, "language": "Mixed"})

    synthetic_df = pd.DataFrame(rows)
    print(f"Synthetic messages : {len(synthetic_df)}")

    # ── 2. Real Hindi + Telugu from external datasets ─────────────────────────
    print("\nFetching external Hindi/Telugu datasets...")
    external_df = load_external_datasets()

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    df = pd.concat([synthetic_df, external_df], ignore_index=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"Dataset saved -> data/dataset.csv  ({len(df)} total samples)")
    print(f"  Synthetic (UPI-specific) : {len(synthetic_df)}")
    print(f"  External  (real-world)   : {len(external_df)}")
    print(f"\nLabel distribution:")
    print(df["label"].map({0: "Legitimate", 1: "Scam"}).value_counts().to_string())
    print(f"\nLanguage distribution:")
    print(df["language"].value_counts().to_string())
    print(f"\nBreakdown by language + label:")
    print(df.groupby(["language", "label"]).size().to_string())


if __name__ == "__main__":
    generate()
