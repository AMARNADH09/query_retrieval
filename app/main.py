# main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import os, mimetypes, tempfile, requests
from dotenv import load_dotenv

load_dotenv(override=True)

API_PREFIX = "/api/v1"
TEAM_TOKEN = os.getenv("TEAM_TOKEN")
app = FastAPI(title="HackRx Query-Retrieval")

security = HTTPBearer(auto_error=True)

class RunReq(BaseModel):
    documents: str
    questions: List[str]

class RunResp(BaseModel):
    answers: List[str]

def _auth(creds: HTTPAuthorizationCredentials):
    if not TEAM_TOKEN:
        raise HTTPException(500, "TEAM_TOKEN not set")
    if creds.scheme.lower() != "bearer":
        raise HTTPException(401, "Invalid auth scheme")
    if creds.credentials.strip() != TEAM_TOKEN:
        raise HTTPException(403, "Invalid token")

from app.doc_parser import parse_document
from app.pipeline import answer_questions

@app.post(f"{API_PREFIX}/hackrx/run", response_model=RunResp)
def hackrx_run(req: RunReq, creds: HTTPAuthorizationCredentials = Depends(security)):
    _auth(creds)

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

    text, meta = parse_document(path)
    answers = answer_questions(text, req.questions, meta=meta)
    return RunResp(answers=answers)
