from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_user_profile_info():
    """
    Ask user about skills, experience, location prefs, etc.
    Return a single text that we can then embed and do retrieval with.
    """
    skills = input("What are your top skills?\n")
    experience = input("Tell me briefly about your past experience.\n")
    location_pref = input("Any location preference?\n")
    # You can keep going with more questions as needed
    
    user_profile_text = f"""
    Skills: {skills}
    Experience: {experience}
    Location Preference: {location_pref}
    """
    return user_profile_text.strip()

def embed_text(text):
    """
    Use your favorite embedding model/LLM to embed the text.
    Could be a local function, or call an external API (OpenAI, etc.)
    """
    logging.info("Embedding the user profile text...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Loaded once globally, ideally
    vector = model.encode(text)
    return vector.tolist()

def retrieve_relevant_jobs(cursor, user_embedding, top_k=5):
    """
    Query your EVA table to get the top_k similar job postings 
    based on the user_embedding.
    """
    # Suppose you have a single table `palantir_jobs` with columns:
    # job_id, job_title, embedding
    # Example EVA SQL:
    logging.info(f"Retrieving top {top_k} relevant jobs from EVA...")
    
    cursor.query(f"""
    SELECT job_id, job_title
    FROM palantir_jobs
    ORDER BY Similarity(embedding, {user_embedding})
    LIMIT {top_k};
    """).df()
    results = cursor.fetchall()
    
    # Format results into a list of dicts for convenience
    top_jobs = []
    for row in results:
        top_jobs.append({
            "job_id": row[0],
            "job_title": row[1]
        })
    return top_jobs

def sanitize_eva_string(input_str: str) -> str:
    """
    Replace single quotes with double single-quotes
    and remove newline characters, which can break EVA's parser.
    """
    # Replace any single quote ' with ''
    sanitized = input_str.replace("'", "''")
    # Replace or remove newline characters
    sanitized = sanitized.replace("\n", " ")
    return sanitized

def make_eva_array_literal(vector: list[float]) -> str:
    """
    Given a Python list of floats, convert it to a string like:
    "ARRAY(0.1, -0.05, 0.48, ...)" for EVA queries.
    """
    # Format each float to a safe string, e.g. "0.123456"
    items_str = ",".join(f"{x:.6f}" for x in vector)
    return f"ARRAY({items_str})"

def job_match_retrieval(cursor, user_profile_text: str, doc_name: str, limit: int = 3):
    """
    1. Sanitize the user's text
    2. Use EVA's SentenceFeatureExtractor(...) for embedding
    3. Select top 'limit' rows ordered by the inline Similarity(...) call
    4. No "AS simscore" alias is used
    """
    # 1) Sanitize text
    safe_text = sanitize_eva_string(user_profile_text)

    # 2) Build the query WITHOUT an alias
    #    We'll do:
    #      SELECT columns, Similarity(...) 
    #      ORDER BY Similarity(...)
    #    That yields an unnamed similarity column in the final df.
    query = f"""
        SELECT
            job_id,
            job_title,
            department,
            location,
            workplace_type,
            data,
            Similarity(SentenceFeatureExtractor('{safe_text}'), features)
        FROM {doc_name}_features
        ORDER BY Similarity(SentenceFeatureExtractor('{safe_text}'), features)
        LIMIT {limit};
    """

    # 3) Execute
    try:
        df = cursor.query(query).df()
    except Exception as e:
        print(f"[ERROR] Could not query table '{doc_name}_features': {e}")
        return []

    # 4) Convert each row to a dict
    #    The "Similarity(...)" column is presumably the last column in df.columns
    #    We'll store that in something like row_dict["similarity"].
    results = []
    if not df.empty:
        sim_col_name = df.columns[-1]  # The last column is "Similarity(SentenceFeatureExtractor(...), features)"

        for _, row in df.iterrows():
            row_dict = {
                "doc_name":        doc_name,
                "job_id":          row.get("job_id"),
                "job_title":       row.get("job_title"),
                "department":      row.get("department"),
                "location":        row.get("location"),
                "workplace_type":  row.get("workplace_type"),
                "data":            row.get("data"),
                "similarity":      row.get(sim_col_name),  # read from the unnamed column
            }
            results.append(row_dict)

    return results

def aggregate_job_matches(cursor, doc_names, user_profile_text, per_table_limit=3, global_top_k=5):
    all_matches = []
    for doc_name in doc_names:
        partial = job_match_retrieval(cursor, user_profile_text, doc_name, limit=per_table_limit)
        all_matches.extend(partial)

    # Suppose "similarity" is bigger => more similar. Then sort descending:
    all_matches.sort(key=lambda x: x["similarity"], reverse=False)

    # Keep top global_top_k
    return all_matches[:global_top_k]
