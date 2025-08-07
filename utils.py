import os
import pinecone
from openai import OpenAI
from pinecone import ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY)

# Ensure the index exists
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

index = pinecone.Index(PINECONE_INDEX_NAME)

client = OpenAI(api_key=OPENAI_API_KEY)

def ingest_pdf_to_pinecone(texts):
    for i, text in enumerate(texts):
        index.upsert([(str(i), text["embedding"], {"text": text["text"]})])

def query_pinecone_with_gpt(query):
    # Get query embedding
    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = embedding_response.data[0].embedding

    # Query Pinecone
    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Extract context from top results
    context = "\n".join([match["metadata"]["text"] for match in result.matches])

    # Use GPT to answer based on context
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return completion.choices[0].message.content
