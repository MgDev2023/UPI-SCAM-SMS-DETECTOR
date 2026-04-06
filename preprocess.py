import re


def preprocess_text(text: str) -> str:
    """
    Clean SMS text while preserving 8 Indian language Unicode blocks
    (Devanagari, Bengali, Gujarati, Gurmukhi, Tamil, Telugu, Kannada, Malayalam).
    Lowercases English, collapses URLs to token, strips punctuation/whitespace.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Replace URLs with a token so the feature is retained
    text = re.sub(r"http\S+|www\.\S+|bit\.ly/\S+", " urltoken ", text)

    # Keep: all Indian language Unicode blocks + English word-chars + spaces
    # Devanagari (Hindi/Marathi), Bengali, Gujarati, Gurmukhi,
    # Tamil, Telugu, Kannada, Malayalam
    text = re.sub(
        r"[^\u0900-\u097F"   # Devanagari  — Hindi, Marathi
        r"\u0980-\u09FF"     # Bengali
        r"\u0A80-\u0AFF"     # Gujarati
        r"\u0A00-\u0A7F"     # Gurmukhi (Punjabi)
        r"\u0B80-\u0BFF"     # Tamil
        r"\u0C00-\u0C7F"     # Telugu
        r"\u0C80-\u0CFF"     # Kannada
        r"\u0D00-\u0D7F"     # Malayalam
        r"\w\s]", " ", text
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
