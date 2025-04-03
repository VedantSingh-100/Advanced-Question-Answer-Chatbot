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
from vector_store import generate_vector_stores, generate_unified_vector_store
from retrieval import vector_retrieval, summary_retrieval
from aggregator import response_aggregator
from job_seeking import get_user_profile_info, embed_text, retrieve_relevant_jobs, aggregate_job_matches 


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
    # vector_stores = generate_vector_stores(cursor, palantir_docs)
    vector_stores = generate_unified_vector_store(cursor, palantir_docs)

    # 3. A user task describing the system context
    user_task = """We have a database of job postings from Palantir.
                   We are building an application to answer questions about these jobs.
                   The documents are each representing a single job with fields like job title, location, etc."""
    
    llm_model = "gpt-3.5-turbo"

    total_cost = 0
    while True:
        question_cost = 0
        # Get question from user
        user_input = input("Hi! Do you want (1) relevant job matches OR (2) general job questions? (Type 'exit' to quit)\n").lower()
        if user_input == "exit":
            break
        if user_input in ["1", "job matches", "matches"]:
            user_profile_text = get_user_profile_info()
            all_matches = aggregate_job_matches(
                cursor=cursor,
                doc_names=doc_names,
                user_profile_text=user_profile_text,
                per_table_limit=3,  # top 3 from each file
                global_top_k=5      # then pick best 5 overall
            )
            if not all_matches:
                print("\n[INFO] No matches found.\n")
            else:
                print("\nTop Job Matches (aggregated):\n")
                for match in all_matches:
                    print(
                        f"Doc Name: {match['doc_name']}\n"
                        f"Job ID: {match['job_id']}\n"
                        f"Title: {match['job_title']}\n"
                        f"Department: {match['department']}\n"
                        f"Location: {match['location']}\n"
                        f"Workplace Type: {match['workplace_type']}\n"
                        f"---"
                    )
        elif user_input in ["2", "general questions", "questions"]:
            question = str(input("Question (enter 'exit' to exit): "))
            if question.lower() == "exit":
                break
            subquestions_bundle_list, cost = generate_subquestions(
                question=question,
                file_names=doc_names,  # e.g. ["PALANTIR_JOBS_1", ...] or ["ALL_JOBS"]
                user_task=user_task,
                llm_model=llm_model,
            )
            question_cost += cost
            responses = []

            for q_no, item in enumerate(subquestions_bundle_list):
                subquestion = item.question
                selected_func = item.function.value

                # If the LLM provided multiple file_names in the subquestion:
                for doc_enum in item.file_names:
                    selected_doc = doc_enum.value

                    if selected_func == "vector_retrieval":
                        response, retrieval_cost = vector_retrieval(
                            cursor, llm_model, subquestion, selected_doc
                        )
                        question_cost += retrieval_cost
                        print(f"✅ Response from vector retrieval: {response}...")
                        responses.append(response)
                    else:
                        # For example, if your code has another path:
                        # response, cost = summary_retrieval(llm_model, subquestion, ...)
                        # question_cost += cost
                        # responses.append(response)
                        pass

            aggregated_response, cost = response_aggregator(llm_model, question, responses)
            question_cost += cost
            print(f"\n✅ Final response: {aggregated_response}")
            question_cost += cost
            total_cost += question_cost

if __name__ == "__main__":
    main()
