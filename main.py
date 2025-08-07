from fastapi import FastAPI, Request
from utils import ingest_pdf_to_pinecone, query_pinecone_with_gpt
import os

app = FastAPI()

@app.post("/hackrx/run")
async def run_rag(request: Request):
    data = await request.json()
    user_query = data.get("query")

    if not user_query:
        return {"error": "Query field is missing"}

    response = query_pinecone_with_gpt(user_query)
    return {"response": response}
