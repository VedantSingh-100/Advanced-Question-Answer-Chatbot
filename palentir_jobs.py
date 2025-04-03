import re
from bs4 import BeautifulSoup

def clean_html(html_content):
    """
    Remove HTML tags from a string and return clean text.
    """
    soup = BeautifulSoup(html_content or "", "html.parser")
    # Convert <li> to line-broken items, etc. 
    # You can refine this based on how you want bullet points formatted.
    text = soup.get_text(separator="\n", strip=True)
    # Optionally do further formatting or whitespace cleanup:
    text = re.sub(r"\n\s*\n+", "\n\n", text.strip())  # condense consecutive newlines
    return text

def transform_job_posting(posting_dict):
    """
    Extracts the most relevant job data for an LLM-based RAG pipeline
    and returns a single textual representation.
    """
    # [Extract fields as before...]
    job_id       = posting_dict.get("id", "")
    job_title    = posting_dict.get("text", "")
    country      = posting_dict.get("country", "")
    workplace    = posting_dict.get("workplaceType", "")

    cat          = posting_dict.get("categories", {})
    commitment   = cat.get("commitment", "")
    department   = cat.get("department", "")
    level        = cat.get("level", "")
    location     = cat.get("location", "")
    team         = cat.get("team", "")
    all_locations= cat.get("allLocations", [])  # array of location strings

    tags_list    = posting_dict.get("tags", [])  # e.g. ['Software', 'Entry Level']
    content_obj  = posting_dict.get("content", {})
    desc_html    = content_obj.get("descriptionHtml", "")
    closing_html = content_obj.get("closingHtml", "")
    lists_info   = content_obj.get("lists", [])

    description_text = clean_html(desc_html)

    bullet_section_text = []
    for section in lists_info:
        heading   = section.get("text", "")
        body_html = section.get("content", "")
        body_txt  = clean_html(body_html)
        if heading or body_txt:
            bullet_section_text.append(f"{heading}\n{body_txt}")

    closing_text = clean_html(closing_html)
    tags_str   = ", ".join(tags_list)
    all_loc_str= ", ".join(all_locations)

    final_doc = {
"job_id": {job_id},
"job_title": {job_title},
"commitment": {commitment},
"department": {department},
"team": {team},
"level": {level},
"location": {location},
"all_locations": {all_loc_str},
"country": {country},
"workplace_type": {workplace},
"tags": {tags_str},
"description":{description_text},
"bullet_sections":{('\n\n'.join(bullet_section_text))},
"closing_text":{closing_text.strip()}}
    return final_doc


import os
import requests

def scrape_palantir_jobs(url=None):
    """
    Simple GET call to the Palantir (or Lever-based) jobs API.
    Returns a list of raw posting dictionaries.
    """
    if url is None:
        url = "https://www.palantir.com/api/lever/v1/postings?state=published&offset=%5B1696975658415%2C%2264602c2e-4581-46eb-822d-e2172ee85937%22%5D"

    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    # Often the JSON might be a dict with a 'postings' key or it might be a list.
    # Adapt accordingly:
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    elif isinstance(data, list):
        return data
    else:
        return []

# def chunk_text_and_attach_metadata(post_dict):
#     """
#     1) Gather metadata from post_dict (transform_job_posting).
#     2) Create one big text from the final_doc (already done by transform).
#        But we'll chunk the *already combined* text if we wish, or we can chunk
#        only the "description" portion. This example lumps everything into one big text.
#     3) Return a list of row dicts, each row containing chunked text + metadata columns.
#     """
#     # Step 1: get all fields as a dictionary
#     meta = transform_job_posting(post_dict)

#     # Combine the textual fields if you want to chunk the entire "final doc".
#     # We'll do that below: 'final_doc' is basically all text fields concatenated.
#     final_doc = (
#         f"JOB TITLE: {meta['job_title']}\n"
#         f"DEPARTMENT: {meta['department']}\n"
#         f"LEVEL: {meta['level']}\n"
#         f"LOCATION: {meta['location']}\n"
#         f"TEAM: {meta['team']}\n"
#         f"ALL_LOCATIONS: {meta['all_locations']}\n"
#         f"WORKPLACE_TYPE: {meta['workplace_type']}\n"
#         f"TAGS: {meta['tags']}\n\n"
#         f"DESCRIPTION:\n{meta['description']}\n\n"
#         f"BULLET SECTIONS:\n{meta['bullet_sections']}\n\n"
#         f"CLOSING:\n{meta['closing_text']}"
#     )

#     # Optionally collapse repeated whitespace in final_doc
#     final_doc = " ".join(final_doc.split())

#     # Step 2: chunk the final_doc
#     chunks = sentence_aware_chunker(final_doc, max_chunk_size=500)

#     # Build row-dicts, each containing all meta + chunked text + chunk_id
#     rows = []
#     for idx, c in enumerate(chunks):
#         # Copy the entire metadata dictionary so each row has all the fields
#         row = dict(meta)
#         # Overwrite or add fields specific to the chunk
#         row["chunk_id"] = idx
#         row["data"] = c
#         rows.append(row)

#     return rows

import re

def sentence_aware_chunker(text, max_chunk_size=500):
    """
    1) Split text into sentences using a simple regex that looks for '.', '?', '!' followed by space/newline.
    2) Accumulate sentences in a buffer until the total length would exceed max_chunk_size.
    3) Yield that chunk, then continue.
    """
    # A very naive sentence split on punctuation followed by whitespace or the end of text
    # (You might replace this with a more advanced library like NLTK or spaCy.)
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # If adding this sentence would exceed the chunk size, finalize the current chunk
        if current_len + sent_len > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = []
            current_len = 0
        
        current_chunk.append(sent)
        current_len += sent_len + 1  # +1 for space/punctuation

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks

import csv

# def write_job_chunks_to_csv(filename, row_dicts):
#     if not row_dicts:
#         return
#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     field_names = list(row_dicts[0].keys())
#     with open(filename, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=field_names)
#         writer.writeheader()
#         for row in row_dicts:
#             writer.writerow(row)

# def load_palantir_job_postings(postings):
#     """
#     postings is the list of raw job data.
#     This function will:
#       - chunk each job's text
#       - write CSV files for each job
#     Returns doc_names, i.e. the list of table/base file names
#     """
#     doc_names = []
#     for i, post_dict in enumerate(postings, start=1):
#         doc_name = f"PALANTIR_JOBS_{i}"
#         # chunk + metadata
#         rows = chunk_text_and_attach_metadata(post_dict)
#         # write CSV
#         file_path = f"data/palantir_careers/{doc_name}.csv"
#         write_job_chunks_to_csv(file_path, rows)
#         doc_names.append(doc_name)
#     return doc_names

def chunk_text_and_attach_metadata(post_dict, doc_name):
    """
    1) Gather metadata from post_dict via transform_job_posting.
    2) Create one big text from those fields.
    3) Chunk the combined text.
    4) Return a list of row dicts, each row containing:
        - all metadata columns from 'meta'
        - doc_name (the CSV identifier)
        - chunk_id
        - data (the chunked text)
    """
    meta = transform_job_posting(post_dict)

    # Combine textual fields if desired
    final_doc = (
        f"JOB TITLE: {meta['job_title']}\n"
        f"DEPARTMENT: {meta['department']}\n"
        f"LEVEL: {meta['level']}\n"
        f"LOCATION: {meta['location']}\n"
        f"TEAM: {meta['team']}\n"
        f"ALL_LOCATIONS: {meta['all_locations']}\n"
        f"WORKPLACE_TYPE: {meta['workplace_type']}\n"
        f"TAGS: {meta['tags']}\n\n"
        f"DESCRIPTION:\n{meta['description']}\n\n"
        f"BULLET SECTIONS:\n{meta['bullet_sections']}\n\n"
        f"CLOSING:\n{meta['closing_text']}"
    )

    # Optionally collapse repeated whitespace
    final_doc = " ".join(final_doc.split())

    # Chunk the text
    chunks = sentence_aware_chunker(final_doc, max_chunk_size=500)

    # Build row-dicts with all metadata + doc_name + chunked text
    rows = []
    for idx, c in enumerate(chunks):
        # Make a copy of meta so each row has the original fields
        row = dict(meta)
        # Add doc_name, chunk_id, and data
        row["doc_name"] = doc_name
        row["chunk_id"] = idx
        row["data"] = c
        rows.append(row)

    return rows


# ------------------------------------------------------------------
# 2) write_job_chunks_to_csv ensuring doc_name is the FIRST column
# ------------------------------------------------------------------
def write_job_chunks_to_csv(filename, row_dicts):
    """
    Writes row_dicts into a CSV file at 'filename'.
    We ensure 'doc_name' is the FIRST column in the CSV.
    """
    if not row_dicts:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Extract the field names from the first row
    all_fields = list(row_dicts[0].keys())

    # Force doc_name to be the first column if present
    if "doc_name" in all_fields:
        # Remove doc_name then prepend it
        all_fields.remove("doc_name")
        field_names = ["doc_name"] + all_fields
    else:
        field_names = all_fields

    # Write CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for row in row_dicts:
            writer.writerow(row)


# ------------------------------------------------------------------
# 3) load_palantir_job_postings to produce CSVs with doc_name inside
# ------------------------------------------------------------------
def load_palantir_job_postings(postings):
    """
    postings is a list of raw job data.

    For each post_dict:
      - chunk the text (and add doc_name)
      - write out a CSV with doc_name as the first column.

    Returns a list of doc_names (the base file names).
    """
    doc_names = []
    for i, post_dict in enumerate(postings, start=1):
        doc_name = f"PALANTIR_JOBS_{i}"
        # chunk + attach doc_name + metadata
        rows = chunk_text_and_attach_metadata(post_dict, doc_name)

        # write CSV
        file_path = f"data/palantir_careers/{doc_name}.csv"
        write_job_chunks_to_csv(file_path, rows)

        doc_names.append(doc_name)
    return doc_names