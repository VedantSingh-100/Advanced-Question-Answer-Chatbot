import os
from dotenv import load_dotenv
from pathlib import Path
import requests
import sys

import warnings
warnings.filterwarnings("ignore")

from subquestion_generator import generate_subquestions
import evadb
from openai_utils import llm_call
from palentir_jobs import scrape_palantir_jobs,load_palantir_job_postings
from vector_store import generate_vector_stores
from retrieval import vector_retrieval, summary_retrieval
from aggregator import response_aggregator 


if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)
def main():
    cursor = evadb.connect().cursor()

    doc_names = [f"PALANTIR_JOBS_{i}" for i in range(1, 85)]

    postings = scrape_palantir_jobs()
    palantir_docs = load_palantir_job_postings(postings)
    vector_stores = generate_vector_stores(cursor, palantir_docs)

    # 3. A user task describing the system context
    user_task = """We have a database of job postings from Palantir.
                   We are building an application to answer questions about these jobs.
                   The documents are each representing a single job with fields like job title, location, etc."""
    
    llm_model = "gpt-3.5-turbo"

    total_cost = 0
    while True:
        question_cost = 0
        # Get question from user
        question = str(input("Question (enter 'exit' to exit): "))
        if question.lower() == "exit":
            break
        subquestions_bundle_list, cost = generate_subquestions(question=question,
                                                               file_names=doc_names,
                                                               user_task=user_task,
                                                               llm_model=llm_model)
        question_cost += cost
        responses = []
        for q_no, item in enumerate(subquestions_bundle_list):
            subquestion = item.question
            selected_func = item.function.value
            for doc_enum in item.file_names:
                selected_doc = doc_enum.value 

                # Validate doc name
                if selected_doc not in doc_names:
                    print(
                        f"[ERROR] '{selected_doc}' is not in doc_names!\n"
                        f"        We only have: {doc_names[:5]} ... (and so on)"
                    )
                    sys.exit(1)

                # If it's a vector_retrieval function, call your retrieval
                if selected_func == "vector_retrieval":
                    response, retrieval_cost = vector_retrieval(
                        cursor, llm_model, subquestion, selected_doc
                    )
                elif selected_func == "llm_retrieval":
                    response, cost = summary_retrieval(llm_model, subquestion, palantir_docs[selected_doc])
            print(f"✅ Response #{q_no+1}: {response}")
            responses.append(response)
            question_cost += cost

        aggregated_response, cost = response_aggregator(llm_model, question, responses)
        print(f"\n✅ Final response: {aggregated_response}")
        question_cost += cost
        total_cost += question_cost

if __name__ == "__main__":
    main()
