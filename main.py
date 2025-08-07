from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
from typing import List
from utils import extract_text_from_pdf_url, split_text, embed_and_store_chunks, parse_question, search_chunks, generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run_rag_api(data: QueryRequest, authorization: str = Header(...)):
    # Step 1: Load and Extract
    text = extract_text_from_pdf_url(data.documents)

    # Step 2: Chunk and Embed
    chunks = split_text(text)
    embed_and_store_chunks(chunks)

    # Step 3: Process Each Question
    answers = []
    for question in data.questions:
        parsed = parse_question(question)
        matched_chunks = search_chunks(parsed)
        answer = generate_answer(question, matched_chunks)
        answers.append(answer)

    return {"answers": answers}
