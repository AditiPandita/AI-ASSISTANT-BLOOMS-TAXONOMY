from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingestion.detector import detect_file_type
from app.ingestion.router import extract_content
from app.chunking import chunk_pages
from app.embeddings import generate_embeddings
from app.rag import answer_question
from app.store import vector_store


app = FastAPI(
    title="AI Teacher Assistant",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str


@app.get("/api/health")
def health_check():

    return {
        "status": "ok",
        "message": "AI Teacher Assistant backend is running"
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    file_content = await file.read()

    file_info = detect_file_type(
        file.filename,
        file.content_type
    )

    extracted_content = extract_content(
        file_content,
        file_info["extension"]
    )

    pages = extracted_content.get(
        "pages",
        []
    )

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No readable content found in the document."
        )

    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=100
    )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not create chunks from the document."
        )

    embedded_chunks = generate_embeddings(
        chunks
    )

    # Reset the current document's vector store
    vector_store.index.reset()
    vector_store.chunks.clear()

    vector_store.add_chunks(
        embedded_chunks
    )

    return {
        "message": "Document processed successfully.",
        "filename": file.filename,
        "file_size": len(file_content),
        "pages": len(pages),
        "chunks": len(chunks),
        "vectors": vector_store.index.ntotal
    }


@app.post("/api/chat")
def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="Please upload a document before asking questions."
        )

    result = answer_question(
        request.question,
        vector_store,
        top_k=3
    )

    sources = [
        {
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "score": item["score"]
        }
        for item in result["results"]
    ]

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": sources
    }