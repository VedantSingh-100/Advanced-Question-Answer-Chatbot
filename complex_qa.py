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
from palentir_jobs import write_jobs_to_files, scrape_palantir_jobs, transform_job_posting


if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)


# def generate_vector_stores(cursor, docs):
#     """Generate a vector store for the docs using evadb.
#     """
#     for doc in docs:
#         print(f"Creating vector store for {doc}...")
#         cursor.query(f"DROP TABLE IF EXISTS {doc};").df()
#         cursor.query(f"LOAD DOCUMENT 'data/{doc}.txt' INTO {doc};").df()
#         evadb_path = os.path.dirname(evadb.__file__)
#         cursor.query(
#             f"""CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
#             IMPL  '{evadb_path}/functions/sentence_feature_extractor.py';
#             """).df()

#         cursor.query(
#             f"""CREATE TABLE IF NOT EXISTS {doc}_features AS
#             SELECT SentenceFeatureExtractor(data), data FROM {doc};"""
#         ).df()

#         cursor.query(
#             f"CREATE INDEX IF NOT EXISTS {doc}_index ON {doc}_features (features) USING FAISS;"
#         ).df()
#         print(f"Successfully created vector store for {doc}.")

import os
import evadb

def generate_vector_stores(cursor, docs):
    """Generate a vector store for the docs using evadb.
    """
    for doc in docs:
        print(f"Creating vector store for {doc}...")
        cursor.query(f"DROP TABLE IF EXISTS {doc};").df()
        file_path = f"data/palantir_careers/{doc}.txt"
        print(f"Attempting to load from file_path = '{file_path}'")
        cursor.query(f"LOAD DOCUMENT '{file_path}' INTO {doc};").df()
        evadb_path = os.path.dirname(evadb.__file__)
        cursor.query(
            f"""CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
            IMPL  '{evadb_path}/functions/sentence_feature_extractor.py';
            """).df()

        cursor.query(
            f"""CREATE TABLE IF NOT EXISTS {doc}_features AS
            SELECT SentenceFeatureExtractor(data), data FROM {doc};"""
        ).df()

        cursor.query(
            f"CREATE INDEX IF NOT EXISTS {doc}_index ON {doc}_features (features) USING FAISS;"
        ).df()
    
    df_sample = cursor.query("SELECT * FROM PALANTIR_JOBS_1_features LIMIT 1;").df()
    print(df_sample.columns)

def generate_union_features_table(cursor, doc_names):
    """
    Drops any existing union table and creates a new union table that 
    combines all doc-specific feature tables (vertically stacks rows).
    """
    cursor.query("DROP TABLE IF EXISTS palantir_union_features;").df()
    
    # Build the union query parts dynamically
    union_query_parts = []
    for doc in doc_names:
        union_query_parts.append(
            f"SELECT _row_id, data, features, '{doc}' AS doc_source FROM {doc}_features"
        )
    
    # Join all parts with UNION ALL
    union_query = " UNION ALL ".join(union_query_parts)
    
    # Create the unified table
    full_query = f"CREATE TABLE palantir_union_features AS {union_query};"
    result = cursor.query(full_query).df()
    print("Created union table 'palantir_union_features' with doc_source:", result)


# def generate_vector_stores(cursor, doc_names, folder="data/palantir_careers"):
#     """
#     For each document name, load its corresponding .txt file,
#     create a features table, and build a FAISS index.
#     Returns the list of document names (to be used later).
#     """
#     for doc in doc_names:
#         print(f"Creating vector store for {doc}...")
#         # Drop any existing table with this name
#         cursor.query(f"DROP TABLE IF EXISTS {doc};").df()
#         # Load the document (each file is handled separately)
#         cursor.query(f"LOAD DOCUMENT '{folder}/{doc}.txt' INTO {doc};").df()
#         # Create the embedding function if it doesn't exist
#         evadb_path = os.path.dirname(evadb.__file__)
#         cursor.query(f"""
#             CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
#             IMPL  '{evadb_path}/functions/sentence_feature_extractor.py';
#         """).df()
#         # Create a new table with embeddings
#         cursor.query(f"""
#             CREATE TABLE IF NOT EXISTS {doc}_features AS
#             SELECT SentenceFeatureExtractor(data) AS features, data FROM {doc};
#         """).df()
#         # Build the FAISS index on the features
#         cursor.query(f"""
#             CREATE INDEX IF NOT EXISTS {doc}_index
#             ON {doc}_features (features)
#             USING FAISS;
#         """).df()
#         print(f"Successfully created vector store for {doc}.")
#     return doc_names
    # No need to return anything (no list returned).

    # Step 3: For each document, create a corresponding vector store
    # for doc in doc_names:
    #     # Sanitize the document name by replacing hyphens with underscores.
    #     sanitized_doc = doc.replace('-', '_')
        
    #     # Drop the table if it exists and load the document into a new table.
    #     cursor.query(f"DROP TABLE IF EXISTS {sanitized_doc};").df()
    #     cursor.query(f"LOAD DOCUMENT 'data/palantir_careers/{doc}.txt' INTO {sanitized_doc};").df()

    #     # Create the embedding function if it doesn't exist.
    #     evadb_path = os.path.dirname(evadb.__file__)
    #     cursor.query(f"""
    #         CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
    #         IMPL '{evadb_path}/functions/sentence_feature_extractor.py';
    #     """).df()

    #     # Create the features table for the document.
    #     features_table = f"{sanitized_doc}_features"
    #     cursor.query(f"""
    #         CREATE TABLE IF NOT EXISTS {features_table} AS
    #         SELECT SentenceFeatureExtractor(data), data
    #         FROM {sanitized_doc};
    #     """).df()

    #     # Create the FAISS index on the features table.
    #     cursor.query(f"""
    #         CREATE INDEX IF NOT EXISTS {sanitized_doc}_index
    #         ON {features_table} (features) USING FAISS;
    #     """).df()
    
    # print(f"[INFO] Created vector stores for {len(doc_names)} job docs.")
    # return doc_names

# def vector_retrieval(cursor, llm_model, question, doc_name):
#     """Returns the answer to a factoid question using vector retrieval.
#     """
#     res_batch = cursor.query(
#         f"""SELECT data FROM {doc_name}_features
#         ORDER BY Similarity(SentenceFeatureExtractor('{question}'),features)
#         LIMIT 3;"""
#     ).df()
#     context_list = []
#     for i in range(len(res_batch)):
#         context_list.append(res_batch["data"][i])
#     context = "\n".join(context_list)
#     user_prompt = f"""You are an assistant for question-answering tasks.
#                 Use the following pieces of retrieved context to answer the question.
#                 If you don't know the answer, just say that you don't know.
#                 Use three sentences maximum and keep the answer concise.
#                 Question: {question}
#                 Context: {context}
#                 Answer:"""

#     response, cost = llm_call(model=llm_model, user_prompt=user_prompt)

#     answer = response.choices[0].message.content
#     return answer, cost

def vector_retrieval(cursor, llm_model, question, doc_name):
    """Returns the answer to a factoid question using vector retrieval.
    """
    res_batch = cursor.query(
        f"""SELECT data FROM {doc_name}_features
        ORDER BY Similarity(SentenceFeatureExtractor('{question}'),features)
        LIMIT 3;"""
    ).df()
    context_list = []
    for i in range(len(res_batch)):
        context_list.append(res_batch["data"][i])
    context = "\n".join(context_list)
    user_prompt = f"""You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Use three sentences maximum and keep the answer concise.
                Question: {question}
                Context: {context}
                Answer:"""

    response, cost = llm_call(model=llm_model, user_prompt=user_prompt)

    answer = response.choices[0].message.content
    return answer, cost

def summary_retrieval(llm_model, question, doc):
    """Returns the answer to a summarization question over the document using summary retrieval.
    """
    # context_length = OPENAI_MODEL_CONTEXT_LENGTH[llm_model]
    # total_tokens = get_num_tokens_simple(llm_model, wiki_docs[doc])
    user_prompt = f"""Here is some context: {doc}
                Use only the provided context to answer the question.
                Here is the question: {question}"""

    response, cost = llm_call(model=llm_model, user_prompt=user_prompt)
    answer = response.choices[0].message.content
    return answer, cost
    # load max of context_length tokens from the document


def response_aggregator(llm_model, question, responses):
    """Aggregates the responses from the subquestions to generate the final response.
    """
    print("-------> ⭐ Aggregating responses...")
    system_prompt = """You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Use three sentences maximum and keep the answer concise."""

    context = ""
    for i, response in enumerate(responses):
        context += f"\n{response}"

    user_prompt = f"""Question: {question}
                      Context: {context}
                      Answer:"""

    response, cost = llm_call(model=llm_model, system_prompt=system_prompt, user_prompt=user_prompt)
    answer = response.choices[0].message.content
    # answer = response.generated_text

    return answer, cost


def load_wiki_pages(page_titles=["Toronto", "Chicago", "Houston", "Boston", "Atlanta"]):

    # Download all wiki documents
    for title in page_titles:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                # 'exintro': True,
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page["extract"]

        data_path = Path("data")
        if not data_path.exists():
            Path.mkdir(data_path)

        with open(data_path / f"{title}.txt", "w") as fp:
            fp.write(wiki_text)

    # Load all wiki documents
    city_docs = {}
    for wiki_title in page_titles:
        input_text = open(f"data/{wiki_title}.txt", "r").read()
        city_docs[wiki_title] = input_text[:10000]
    return city_docs

def load_job_docs(doc_names):
    """
    Read each .txt file from data/palantir_careers into a dictionary:
    { doc_name: entire_text, ... }
    """
    job_docs = {}
    for name in doc_names:
        path = os.path.join("data/palantir_careers", f"{name}.txt")
        with open(path, "r", encoding="utf-8") as fp:
            job_docs[name] = fp.read()
    return job_docs

def load_palantir_job_postings(postings, output_dir="data/palantir_careers"):
    """
    1. Writes job postings to files (via write_jobs_to_files).
    2. Reads those files back into a dictionary mapping doc_name -> text.
    3. Returns that dictionary.
    """

    # Step 1: Write postings to disk -> get doc names
    doc_names = write_jobs_to_files(postings, output_dir=output_dir)

    # Step 2: Read them back into a dict
    docs_dict = {}
    for name in doc_names:
        file_path = os.path.join(output_dir, f"{name}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            # You could optionally do text[:10000] to mimic the 10k truncation
            # that was happening in the original code.
            docs_dict[name] = text

    return docs_dict

if __name__ == "__main__":

    # 1. Connect to EvaDB
    print("⏳ Connect to EvaDB...")
    cursor = evadb.connect().cursor()
    print("✅ Connected to EvaDB...")

    doc_names = [f"PALANTIR_JOBS_{i}" for i in range(1, 85)]

    postings = scrape_palantir_jobs()
    palantir_docs = load_palantir_job_postings(postings)
    # 2. Build vector stores for your Palantir job postings
    vector_stores = generate_vector_stores(cursor, palantir_docs)  # returns list of doc names
    # If you want summarization, load all job text in memory
    # job_docs = load_job_docs(vector_stores)

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
        print("🧠 Generating subquestions...")
        subquestions_bundle_list, cost = generate_subquestions(question=question,
                                                               file_names=doc_names,
                                                               user_task=user_task,
                                                               llm_model=llm_model)
        question_cost += cost
        print("\n[DEBUG] Subquestions received from generate_subquestions:")
        for idx, subq_bundle in enumerate(subquestions_bundle_list, start=1):
            print(
                f"  #{idx} => subquestion: '{subq_bundle.question}', "
                f"function: '{subq_bundle.function.value}', "
                f"doc name: '{subq_bundle.file_name.value}'"
            )
        print("-" * 60)
        responses = []
        for q_no, item in enumerate(subquestions_bundle_list):
            subquestion = item.question
            selected_func = item.function.value
            selected_doc = item.file_name.value
            print(f"\n-------> 🤔 Processing subquestion #{q_no+1}: {subquestion} | function: {selected_func} | data source: {selected_doc}")
            if selected_func == "vector_retrieval":
                if selected_doc not in doc_names:
                    print(f"[ERROR] '{selected_doc}' is not in doc_names!\n"
                          f"        We only have: {doc_names[:5]} ... (and so on)")
                    sys.exit(1)
                response, cost = vector_retrieval(cursor, llm_model, subquestion, selected_doc)
            elif selected_func == "llm_retrieval":
                response, cost = summary_retrieval(llm_model, subquestion, palantir_docs[selected_doc])
            else:
                print(f"\nCould not process subquestion: {subquestion} function: {selected_func} data source: {selected_doc}\n")
                exit(0)
            print(f"✅ Response #{q_no+1}: {response}")
            responses.append(response)
            question_cost += cost

        aggregated_response, cost = response_aggregator(llm_model, question, responses)
        question_cost += cost
        print(f"\n✅ Final response: {aggregated_response}")
        print(f"🤑 Total cost for the question: ${question_cost:.4f}")
        total_cost += question_cost

    print(f"Total cost for all questions: ${total_cost:.4f}")

