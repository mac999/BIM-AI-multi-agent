from pathlib import Path

from fastapi import FastAPI, UploadFile, File

from agents.rag_agent import RAGAgent
from orchestrator.orchestrator import Orchestrator
from utils.faiss_utils import build_faiss_index

app = FastAPI(title="BIM AI Assistant")

INDEX_PATH = "data/faiss.index"
DOC_DIR = "data/documents"

# build index at startup
DOCS = build_faiss_index(DOC_DIR, INDEX_PATH)
RAG = RAGAgent(INDEX_PATH, DOCS)
ORCH = Orchestrator(RAG)


@app.post("/ask")
async def ask(question: str):
    return ORCH.handle_question(question)


@app.post("/ask_ifc")
async def ask_ifc(question: str, file: UploadFile = File(...)):
    file_path = Path("data") / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return ORCH.handle_question(question, ifc_path=str(file_path))
