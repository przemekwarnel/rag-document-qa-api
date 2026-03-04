from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from typing import Optional
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline
from rag.ingestion import build_index


load_dotenv()

PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_index")
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1500"))

rag: Optional[RAGPipeline] = None
def load_pipeline():
    global rag
    rag = RAGPipeline(persist_dir=PERSIST_DIR, k=TOP_K, max_context_chars=MAX_CONTEXT_CHARS)

app = FastAPI(title="RAG Bot")


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(request: AskRequest):
    if rag is None:
        if not os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
            raise HTTPException(status_code=409, detail="No index found. Call /ingest first")
        load_pipeline()
    result = rag.answer_question(request.query) # type: ignore
    return result


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    global rag
    rag = None

    if file.filename is None:
        raise HTTPException(status_code=400, detail="Missing filename")
    
    filename = file.filename

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    file.file.close()

    # Rebuild index
    tmp_dir = PERSIST_DIR + "_tmp"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    build_index(file_path, tmp_dir)

    # Move tmp dir to main persist dir
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    shutil.move(tmp_dir, PERSIST_DIR)

    # Reload pipeline after new index
    load_pipeline()

    return {"status": "ok", "filename": file.filename}