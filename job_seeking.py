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

def retrieve_relevant_jobs(cursor, user_profile_text, top_k=5):
    """
    If you do NOT have a 'features' column in all_jobs_features,
    but instead rely on computing embeddings on the fly,
    do something like: 
      ORDER BY Similarity(SentenceFeatureExtractor('{text}'), SentenceFeatureExtractor(data))
    """
    safe_text = sanitize_eva_string(user_profile_text)
    logging.info(f"Retrieving top {top_k} relevant jobs from EVA...")

    query = f"""
        SELECT
            doc_name,
            job_id,
            job_title,
            department,
            location,
            workplace_type,
            data,
            Similarity(
                SentenceFeatureExtractor('{safe_text}'),
                SentenceFeatureExtractor(data)
            ) as sim
        FROM all_jobs_features
        ORDER BY sim DESC
        LIMIT {top_k};
    """

    logging.debug(f"[DEBUG] retrieve_relevant_jobs query:\n{query}")
    df = cursor.query(query).df()
    logging.info(f"[INFO] Rows returned: {len(df)}")

    results = []
    if not df.empty:
        for _, row in df.iterrows():
            row_dict = {
                "doc_name":        row.get("doc_name"),
                "job_id":          row.get("job_id"),
                "job_title":       row.get("job_title"),
                "department":      row.get("department"),
                "location":        row.get("location"),
                "workplace_type":  row.get("workplace_type"),
                "data":            row.get("data"),
                "similarity":      row.get("sim"),
            }
            results.append(row_dict)

    return results


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

def job_match_retrieval(cursor, user_profile_text: str, limit: int = 3):
    """
    Similar to retrieve_relevant_jobs, but we embed the user text inline
    using SentenceFeatureExtractor. This queries the single table `all_jobs_features`.
    """
    safe_text = sanitize_eva_string(user_profile_text)

    query = f"""
        SELECT
            doc_name,
            job_id,
            job_title,
            department,
            location,
            workplace_type,
            data,
            Similarity(
                SentenceFeatureExtractor('{safe_text}'),
                SentenceFeatureExtractor(data)
            ) as sim
        FROM all_jobs_features
        ORDER BY sim DESC
        LIMIT {limit};
    """

    print("[DEBUG] job_match_retrieval() query:\n", query)
    df = cursor.query(query).df()
    print(f"[DEBUG] Rows returned: {len(df)}")

    results = []
    if not df.empty:
        for _, row in df.iterrows():
            row_dict = {
                "doc_name":        row.get("doc_name"),
                "job_id":          row.get("job_id"),
                "job_title":       row.get("job_title"),
                "department":      row.get("department"),
                "location":        row.get("location"),
                "workplace_type":  row.get("workplace_type"),
                "data":            row.get("data"),
                "similarity":      row.get("sim"),
            }
            results.append(row_dict)

    return results

def aggregate_job_matches(cursor, user_profile_text, limit=5):
    """
    Retrieves matches from the single table all_jobs_features
    and returns the top 'limit' rows (already sorted by similarity).
    """
    all_matches = job_match_retrieval(cursor, user_profile_text, limit=limit)

    # If you prefer to re-sort or do a second pass:
    all_matches.sort(key=lambda x: x["similarity"], reverse=True)
    return all_matches[:limit]

