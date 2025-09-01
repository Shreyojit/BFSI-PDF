import uvicorn
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chatbot.config import get_settings
from chatbot.textract_utils import upload_to_s3, start_textract_analysis, wait_for_job, fetch_all_results
from chatbot.parse_textract import parse_blocks
from chatbot.indexing import build_items, build_faiss, load_faiss
from chatbot.qa import ALLOWED_QUESTIONS, search, answer_with_claude
from chatbot.models import AskRequest, AskResponse, Source, IngestResponse

S = get_settings()
faiss_index = None
meta_items: List[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global faiss_index, meta_items
    idx, meta = load_faiss()
    if idx and meta:
        faiss_index, meta_items = idx, meta
        print("FAISS index loaded with", len(meta_items), "items")
    else:
        print("⚠️ No FAISS index loaded")
    yield


app = FastAPI(
    title="STR Intelligence API",
    description="STR Intelligence API for data processing and analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the STR Intelligence API"}


@app.get("/chatbot/health")
def health():
    return {"status": "ok", "index_loaded": faiss_index is not None, "index_items": len(meta_items or [])}


@app.get("/chatbot/questions")
def questions():
    return {"count": len(ALLOWED_QUESTIONS), "questions": ALLOWED_QUESTIONS}


@app.post("/chatbot/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    # 1) upload to S3 (directly from request bytes; no local temp)
    content = await file.read()
    s3_uri = upload_to_s3(content, file.filename)

    # derive key from URI: s3://bucket/key -> key
    prefix = f"s3://{S.AWS_S3_BUCKET_NAME}/"
    if not s3_uri.startswith(prefix):
        raise HTTPException(status_code=500, detail="Unexpected S3 URI format.")
    s3_key = s3_uri[len(prefix):]

    # 2) Textract
    job_id = start_textract_analysis(S.AWS_S3_BUCKET_NAME, s3_key)
    wait_for_job(job_id)
    blocks, meta = fetch_all_results(job_id)
    pages = meta.get("Pages", 0)

    # 3) Parse → items
    lines_by_page, forms_by_page, tables_by_page = parse_blocks(blocks)
    items = build_items(lines_by_page, forms_by_page, tables_by_page)

    # 4) Build FAISS (in-memory if DISABLE_LOCAL_PERSIST=1)
    global faiss_index, meta_items
    faiss_index, meta_items = build_faiss(items)

    return IngestResponse(s3_uri=s3_uri, pages=pages, blocks=len(blocks), index_size=len(meta_items))


@app.post("/chatbot/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.question not in ALLOWED_QUESTIONS:
        raise HTTPException(status_code=400, detail="Question not in allowed set. Call /questions for options.")
    if faiss_index is None or not meta_items:
        raise HTTPException(status_code=409, detail="Index not ready. Ingest a PDF first.")

    hits = search(faiss_index, meta_items, req.question, k=5)
    ans = answer_with_claude(req.question, hits)
    sources = [Source(page=h["page"], type=h["type"], preview=h["text"][:200]) for h in hits]
    return AskResponse(question=req.question, answer=ans, sources=sources)


if __name__ == "__main__":
    uvicorn.run('app:app', host="0.0.0.0", port=8000, reload=True)
