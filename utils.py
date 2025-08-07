import fitz  # PyMuPDF
import requests
from openai import OpenAI
import openai
import pinecone
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Load keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(os.getenv("PINECONE_INDEX"))

# ----------- PDF Extractor -----------
def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("temp.pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ----------- Chunk Text -----------
def split_text(text: str, chunk_size=300) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ----------- Embedding & Store -----------
def embed_and_store_chunks(chunks: List[str]):
    for i, chunk in enumerate(chunks):
        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        vector = response['data'][0]['embedding']
        index.upsert([(f"chunk-{i}", vector, {"text": chunk})])

# ----------- Query Parser -----------
def parse_question(question: str) -> str:
    prompt = f"Extract the core intent and key terms from: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ----------- Search Chunks -----------
def search_chunks(query: str, top_k=3) -> List[str]:
    emb = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    results = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]

# ----------- Answer Generation -----------
def generate_answer(question: str, matched_chunks: List[str]) -> str:
    context = "\n".join(matched_chunks)
    prompt = f"""Answer the question based on the following document context:

Context:
{context}

Question:
{question}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
