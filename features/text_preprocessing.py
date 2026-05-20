"""
Text preprocessing for cyberbullying detection.
Steps: lowercase, URL/mention removal, tokenise, optional stop words/lemmatise.
See practical-approach-implementation-guide.md Step 3.
"""
import re


def lowercase(text):
    return text.lower() if isinstance(text, str) else ""


def remove_urls(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_mentions(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"@\w+", "", text)


def preprocess(text, remove_stop_words=False):
    """Basic pipeline: lowercase, strip URLs and @mentions."""
    t = lowercase(text)
    t = remove_urls(t)
    t = remove_mentions(t)
    t = " ".join(t.split())
    return t
