# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import os, mimetypes, tempfile, requests
from dotenv import load_dotenv

# Load env (handy locally; on Render you set env vars in dashboard)
load_dotenv(override=True)

API_PREFIX = "/api/v1"
TEAM_TOKEN = os.getenv("TEAM_TOKEN") or ""
app = FastAPI(title="HackRx Query-Retrieval")

# Friendly root + health
@app.get("/")
def root():
    return {"ok": True, "service": "HackRx Query-Retrieval", "docs": "/docs"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# Security (Bearer)
security = HTTPBearer(auto_error=True)

class RunReq(BaseModel):
    documents: str
    questions: List[str]

class RunResp(BaseModel):
    answers: List[str]

def _auth(creds: Optional[HTTPAuthorizationCredentials]):
    if not TEAM_TOKEN:
        raise HTTPException(500, "TEAM_TOKEN not set")
    if creds is None:
        raise HTTPException(401, "Missing bearer token")
    if (creds.scheme or "").lower() != "bearer":
        raise HTTPException(401, "Invalid auth scheme")

    sent = (creds.credentials or "").strip()
    expected = TEAM_TOKEN.strip()
    if sent != expected:
        raise HTTPException(403, "Invalid token")

from app.doc_parser import parse_document
from app.pipeline import answer_questions

@app.post(f"{API_PREFIX}/hackrx/run", response_model=RunResp)
def hackrx_run(req: RunReq, creds: HTTPAuthorizationCredentials = Depends(security)):
    _auth(creds)

    # Download the document to a temp file with inferred suffix
    with requests.get(req.documents, stream=True, timeout=45) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").split(";")[0]
        guessed = mimetypes.guess_extension(ctype) or ""
        if not guessed:
            guessed = ".pdf" if req.documents.lower().endswith(".pdf") else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=guessed) as tmp:
            for chunk in r.iter_content(65536):
                if chunk:
                    tmp.write(chunk)
            path = tmp.name

    # Parse and answer
    text, meta = parse_document(path)  # (full_text, meta)
    answers = answer_questions(text, req.questions, meta=meta)
    return RunResp(answers=answers)
