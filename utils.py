import os
import fitz  # PyMuPDF
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to ingest PDF into Pinecone
def ingest_pdf_to_pinecone(pdf_path):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    text = extract_text_from_pdf(pdf_path)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    vectors = []
    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        vectors.append({"id": f"chunk-{i}", "values": embedding, "metadata": {"text": chunk}})

    index.upsert(vectors)

# Function to query Pinecone using GPT

def query_pinecone_with_gpt(query):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    embed = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )

    query_vector = embed.data[0].embedding

    search_results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    context = "\n\n".join([match["metadata"]["text"] for match in search_results.matches])

    prompt = f"Use the following context to answer the question.\nContext:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
