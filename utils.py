import os
import fitz  # PyMuPDF
from openai import OpenAI
import pinecone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")

client = OpenAI(api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Ensure Pinecone index exists
if PINECONE_INDEX not in [index.name for index in pinecone.list_indexes()]:
    pinecone.create_index(PINECONE_INDEX, dimension=1536, metric="cosine")

index = pinecone.Index(PINECONE_INDEX)

def ingest_pdf_to_pinecone(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            chunks.append(text.strip())
    doc.close()

    upserts = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        upserts.append((f"{pdf_path}_{i}", embedding, {"text": chunk}))
    
    index.upsert(vectors=upserts)
    return len(upserts)

def get_embedding(text):
    res = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return res.data[0].embedding

def query_pinecone_with_gpt(query):
    query_embedding = get_embedding(query)
    result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    contexts = [match["metadata"]["text"] for match in result.matches]
    context_text = "\n\n".join(contexts)

    prompt = f"Answer the following question using the provided context:\n\nContext:\n{context_text}\n\nQuestion:\n{query}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
