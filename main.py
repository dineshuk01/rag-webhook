from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from utils import ingest_pdf_to_pinecone, query_pinecone_with_gpt
import os

app = FastAPI()

@app.post("/webhook/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        with open(filename, "wb") as f:
            f.write(content)

        response = ingest_pdf_to_pinecone(filename)
        os.remove(filename)
        return {"message": "File uploaded and ingested", "chunks_uploaded": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/hackrx/run")
async def hackrx_query(query: str = Query(..., description="User's natural language question")):
    try:
        response = query_pinecone_with_gpt(query)
        return {"query": query, "answer": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
