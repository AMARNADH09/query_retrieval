
import re, os

MAX_WORDS = int(os.getenv("MAX_CHUNK_WORDS", "160"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "35"))

def chunk_text(text: str, max_words=MAX_WORDS, overlap=OVERLAP):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur = [], []
    for s in sents:
        words = s.split()
        if cur and len(cur) + len(words) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.extend(words)
    if cur: chunks.append(" ".join(cur))
    return chunks
