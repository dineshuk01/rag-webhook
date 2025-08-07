import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = "us-west-2"  # Update as per your actual region
INDEX_NAME = "rag-index"

# Initialize OpenAI
openai = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone (new SDK)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if not exists
def init_pinecone():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
    return pc.Index(INDEX_NAME)

# Load PDF & split into chunks
def load_pdf_chunks(path: str, chunk_size: int = 300) -> List[str]:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Get OpenAI Embeddings
def get_embeddings(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for chunk in chunks:
        response = openai.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

# Upload to Pinecone index
def upload_to_pinecone(index, chunks: List[str], embeddings: List[List[float]]):
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })
    index.upsert(vectors=vectors)
