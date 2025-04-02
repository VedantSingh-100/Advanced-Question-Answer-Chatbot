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

    final_doc = f"""
JOB ID: {job_id}
TITLE: {job_title}
COMMITMENT: {commitment}
DEPARTMENT: {department}
TEAM: {team}
LEVEL: {level}
LOCATION: {location}
ALL LOCATIONS: {all_loc_str}
COUNTRY: {country}
WORKPLACE TYPE: {workplace}

TAGS: {tags_str}

DESCRIPTION:
{description_text}

BULLET SECTIONS:
{('\n\n'.join(bullet_section_text)).strip()}

CLOSING:
{closing_text}
    """.strip()

    # Collapse all whitespace (including newlines) into single spaces:
    final_doc = " ".join(final_doc.split())
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

# def write_jobs_to_files(postings, output_dir="data/palantir_careers"):
#     """
#     Takes a list of raw job dictionaries,
#     extracts relevant fields, and writes them to .txt files in output_dir.
#     Files are named "PALANTIR_JOBS_1.txt", "PALANTIR_JOBS_2.txt", etc.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for i, posting in enumerate(postings, start=1):
#         job_text = transform_job_posting(posting)
#         # Use our custom naming scheme instead of the job's own id.
#         job_id = f"PALANTIR_JOBS_{i}"
#         out_path = os.path.join(output_dir, f"{job_id}.txt")
#         with open(out_path, "w", encoding="utf-8") as f:
#             f.write(job_text)

#     print(f"[INFO] Wrote {len(postings)} job text files to '{output_dir}'")

def write_jobs_to_files(postings, output_dir="data/palantir_careers"):
    """
    Writes each job posting into a file named PALANTIR_JOBS_{i+1}.txt
    Returns a list of doc names: ["PALANTIR_JOBS_1", "PALANTIR_JOBS_2", ...]
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {output_dir}: {e}")

    doc_names = []
    for i, posting in enumerate(postings):
        doc_name = f"PALANTIR_JOBS_{i+1}"  # e.g., PALANTIR_JOBS_1
        file_path = os.path.join(output_dir, f"{doc_name}.txt")

        # Transform job posting into text
        try:
            job_text = transform_job_posting(posting)
            if not job_text.strip():
                raise ValueError(f"Job text for posting {i+1} is empty.")
        except Exception as e:
            print(f"Error transforming posting {i+1}: {e}")
            continue

        # Write to file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(job_text)
            doc_names.append(doc_name)
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")

    # Validate return value
    assert all(isinstance(name, str) for name in doc_names), "doc_names must contain only strings."
    return doc_names

