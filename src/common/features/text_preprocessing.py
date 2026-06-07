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
    # strip URLs and @mentions before TF-IDF so they don't pollute vocabulary
    t = lowercase(text)
    t = remove_urls(t)
    t = remove_mentions(t)
    t = " ".join(t.split())
    return t


def tokenize(text):
    return preprocess(text).split()
