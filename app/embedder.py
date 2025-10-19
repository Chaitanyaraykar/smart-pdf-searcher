# app/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List

# -----------------------------
# Load Local Embedding Model
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")


# -----------------------------
# STEP 1: Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split long text into overlapping chunks for better embedding quality.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# -----------------------------
# STEP 2: Generate Embeddings
# -----------------------------
def get_embeddings(chunks: List[str]):
    """
    Generate sentence embeddings using a local transformer model.
    """
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


if __name__ == "__main__":
    sample_text = """Artificial Intelligence is revolutionizing industries worldwide.
    Neural networks and transformers are at the heart of this revolution."""
    chunks = chunk_text(sample_text)
    embeddings = get_embeddings(chunks)
    print(f"Created {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

