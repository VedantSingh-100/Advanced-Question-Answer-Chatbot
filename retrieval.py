from openai_utils import llm_call
# def vector_retrieval(cursor, llm_model, question, doc_name):
#     """
#     Returns the answer to a factoid question using vector retrieval,
#     including the metadata in the prompt so the LLM can leverage it.
#     """
#     # 1. Retrieve columns from the database
#     res_batch = cursor.query(
#         f"""
#         SELECT job_id,
#                job_title,
#                commitment,
#                department,
#                team,
#                level,
#                location,
#                all_locations,
#                country,
#                workplace_type,
#                tags,
#                description,
#                bullet_sections,
#                closing_text,
#                chunk_id,
#                data
#         FROM {doc_name}_features
#         ORDER BY Similarity(SentenceFeatureExtractor('{question}'), features)
#         LIMIT 3;
#         """
#     ).df()
    
#     # 2. Build context string that includes metadata
#     context_list = []
#     for i in range(len(res_batch)):
#         row = res_batch.iloc[i]

#         # Construct a metadata "header" to give the LLM helpful context
#         metadata_str = f"""[METADATA]
# Job Title: {row["job_title"]}
# Department: {row["department"]}
# Location: {row["location"]}
# Workplace Type: {row["workplace_type"]}
# Tags: {row["tags"]}

# [DESCRIPTION]
# {row["description"]}

# [BULLET SECTIONS]
# {row["bullet_sections"]}

# [CLOSING TEXT]
# {row["closing_text"]}

# [CHUNK DATA]
# {row["data"]}
#         """

#         context_list.append(metadata_str)

#     # Join all retrieved contexts (from up to 3 top results) into one prompt section
#     context = "\n---\n".join(context_list)

#     # 3. Construct the final prompt for the LLM
#     user_prompt = f"""
# You are an assistant for question-answering tasks.
# Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know.
# Give comprehensive and detailed answers.

# Question: {question}
# Context:
# {context}

# Answer:
# """

#     # 4. Call your LLM (assuming llm_call is your function that wraps the openai API or similar)
#     response, cost = llm_call(model=llm_model, user_prompt=user_prompt)

#     answer = response.choices[0].message.content
#     return answer, cost

def vector_retrieval(cursor, llm_model, question, doc_name=None):
    """
    Returns the answer to a question using vector retrieval from the unified `all_jobs_features` table.
    If doc_name is provided, we filter on that. Otherwise, we search across all doc_names.
    """

    # 2. Construct the query
    #    Ensure 'SentenceFeatureExtractor(data)' is the same column name you used
    #    when creating the FAISS index.
    sql_query = f"""
        SELECT
            doc_name,
            job_id,
            job_title,
            commitment,
            department,
            team,
            level,
            location,
            all_locations,
            country,
            workplace_type,
            tags,
            description,
            bullet_sections,
            closing_text,
            chunk_id,
            data
        FROM all_jobs_features
        ORDER BY
            Similarity(
                SentenceFeatureExtractor('{question}'),
                SentenceFeatureExtractor(data)
            ) DESC
        LIMIT 3;
    """

    # 3. Execute the query
    res_batch = cursor.query(sql_query).df()

    # 4. Build the context string
    context_list = []
    for i in range(len(res_batch)):
        row = res_batch.iloc[i]
        metadata_str = f"""[METADATA]
Source Doc: {row["doc_name"]}
Job Title: {row["job_title"]}
Department: {row["department"]}
Location: {row["location"]}
Workplace Type: {row["workplace_type"]}
Tags: {row["tags"]}

[DESCRIPTION]
{row["description"]}

[BULLET SECTIONS]
{row["bullet_sections"]}

[CLOSING TEXT]
{row["closing_text"]}

[CHUNK DATA]
{row["data"]}
"""
        context_list.append(metadata_str)

    context = "\n---\n".join(context_list)

    # 5. Construct the LLM prompt
    user_prompt = f"""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Give comprehensive and detailed answers.

Question: {question}
Context:
{context}

Answer:
"""

    # 6. Call the LLM
    response, cost = llm_call(model=llm_model, user_prompt=user_prompt)

    # 7. Extract the final answer text
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