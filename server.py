# server.py

import os
import evadb
from fastapi import FastAPI, Body
from typing import List, Optional

from vector_store import table_exists  # or define a helper
from retrieval import vector_retrieval, summary_retrieval
from subquestion_generator import generate_subquestions
from aggregator import response_aggregator
import time

# We'll have a global cursor for reuse
cursor = None

app = FastAPI(
    title="Palantir Jobs Backend",
    description="A robust backend for Palantir jobs Q&A using EvaDB + LLM",
    version="0.1.0"
)

DOC_NAMES = [f"PALANTIR_JOBS_{i}" for i in range(1, 85)]  # or dynamically discovered
LLM_MODEL = "gpt-3.5-turbo"

import time
import logging

logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
def startup_event():
    global cursor
    db_path = "/home/vhsingh/rag-demystified-main/evadb_data"
    start_time = time.perf_counter()
    
    logging.info("Connecting to EvaDB...")
    connection = evadb.connect(db_path)
    logging.info(f"Done connecting to EvaDB in {time.perf_counter() - start_time:.2f} seconds.")

    start_time = time.perf_counter()
    logging.info("Getting cursor...")
    cursor = connection.cursor()
    logging.info(f"Got cursor in {time.perf_counter() - start_time:.2f} seconds.")

    # Next step, checking table existence
    start_time = time.perf_counter()
    logging.info("Checking if table PALANTIR_JOBS_1 exists...")
    if not table_exists(cursor, "PALANTIR_JOBS_1"):
        logging.warning("Table not found. Did you run offline setup?")
    else:
        logging.info("PALANTIR_JOBS_1 found.")
    logging.info(f"Check took {time.perf_counter() - start_time:.2f} seconds.")


@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "OK"}

@app.post("/ask_question")
def ask_question(
    question: str = Body(...),
    doc_name: Optional[str] = Body(None),
    k: int = Body(3)
):
    if cursor is None:
        return {"error": "Cursor not initialized. Check server startup logs."}

    user_task = """We have a database of job postings from Palantir.
                   We are building an application to answer questions about these jobs.
                   The documents are each representing a single job with fields like job title, location, etc."""
    subquestions_list, cost_gs = generate_subquestions(
        question=question, 
        file_names=DOC_NAMES, 
        user_task=user_task,
        llm_model=LLM_MODEL
    )

    question_cost = cost_gs
    responses = []

    # Iterate over each subquestion bundle using dot notation:
    for item in subquestions_list:
        subq = item.question  # use dot notation
        func = item.function   # use dot notation; if you need the string value, use item.function.value
        doc_list = item.file_names  # a list of enums; later you'll extract their .value

        if doc_name:
            doc_list = [doc_name]

        for doc in doc_list:
            selected_doc = doc.value
            if func == "vector_retrieval" or (hasattr(func, "value") and func.value == "vector_retrieval"):
                start_time = time.time()
                resp, retrieval_cost = vector_retrieval(cursor, LLM_MODEL, subq, selected_doc)
                question_cost += retrieval_cost
                responses.append(resp)
            elif func == "llm_retrieval" or (hasattr(func, "value") and func.value == "llm_retrieval"):
                # If you're storing doc text in memory or somewhere
                resp, sum_cost = summary_retrieval(LLM_MODEL, subq, "SOME_DOC_TEXT")
                question_cost += sum_cost
                responses.append(resp)
            else:
                responses.append("Unknown function call.")

    final_answer, agg_cost = response_aggregator(LLM_MODEL, question, responses)
    elapsed = time.time() - start_time
    print(f"The elapsed time is {elapsed}")
    question_cost += agg_cost

    return {
        "final_answer": final_answer,
        "question_cost": question_cost
    }

