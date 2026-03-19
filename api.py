import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ingest import parse_pdf, chunk_pages, build_vectorstore
from rag import load_vectorstore, build_rag_chain

app = FastAPI(title="Legal RAG API")

chain = None

if os.path.exists("./chroma_db"):
    vectorstore = load_vectorstore()
    chain = build_rag_chain(vectorstore)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """uploads PDF and indexes it"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    # saving temporarily
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # indexing
    pages = parse_pdf(tmp_path)
    documents = chunk_pages(pages)
    vectorstore = build_vectorstore(documents)

    # creating chain with new data
    global chain
    chain = build_rag_chain(vectorstore)

    return {"status": "ok", "pages": len(pages), "chunks": len(documents)}


class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    """answer a question about uploaded file"""
    if chain is None:
        raise HTTPException(400, "No documents uploaded yet")

    answer = chain.invoke(request.question)
    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok", "ready": chain is not None}