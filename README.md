
# HackRx LLM-Powered Query–Retrieval (Groq + FAISS)

## 1) Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configure
Copy `.env.example` to `.env` and set:
- `TEAM_TOKEN` (from HackRx portal)
- `GROQ_API_KEY` (your Groq key)
- Optional: `ST_MODEL_DIR` if you pre-download the model

## 3) Run (VS Code or terminal)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Base URL: `http://localhost:8000/api/v1`

## 4) Test
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
 -H "Authorization: Bearer $TEAM_TOKEN" \
 -H "Content-Type: application/json" \
 -d "{\"documents\": \"https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D\", \"questions\": [\"What is the grace period for premium payment?\"] }"
```

## Notes
- Uses Groq chat completions endpoint.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local path if set).
- Retrieval: FAISS IP + MMR + clause-tagged context.
