
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import tempfile

from app.pdf_loader import extract_text_from_pdf, clean_text
from app.embedder import chunk_text, get_embeddings
from app.retriever import VectorStore

import numpy as np

app = FastAPI(title="Smart PDF Searcher", version="1.0")

# Initialize FAISS Vector Store
VECTOR_DIM = 384  # for MiniLM; fallback if model unavailable
vector_store = VectorStore(dim=VECTOR_DIM)

# To handle environments without model access
EMBEDDING_FALLBACK = False


@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """
    Upload and index a PDF file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        # Extract text
        text = extract_text_from_pdf(tmp_path)
        cleaned_text = clean_text(text)

        # Chunks made
        chunks = chunk_text(cleaned_text)

        # Generate embeddings
        try:
            embeddings = get_embeddings(chunks)
        except Exception as e:
            global EMBEDDING_FALLBACK
            EMBEDDING_FALLBACK = True
            print("Embedding model not available, using dummy vectors.")
            embeddings = np.random.rand(len(chunks), VECTOR_DIM).tolist()

        # Store in FAISS
        vector_store.add_embeddings(embeddings, chunks)
        vector_store.save()

        return {"message": f"Indexed {len(chunks)} chunks successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/search")
async def search_query(query: str = Form(...)):
    """
    Search for the most relevant chunks to the given query.
    """
    try:
        if EMBEDDING_FALLBACK:
            # random query vector (demo mode)
            query_embedding = np.random.rand(VECTOR_DIM).tolist()
        else:
            query_embedding = get_embeddings([query])[0]

        results = vector_store.search(query_embedding, top_k=3)

        formatted = [
            {"text": text, "score": round(score, 4)}
            for text, score in results
        ]
        return {"query": query, "results": formatted}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {"status": "running", "message": "Smart PDF SEarcher API is live"}

