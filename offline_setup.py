# main_offline_setup.py

from palentir_jobs import scrape_palantir_jobs, load_palantir_job_postings
from vector_store import generate_vector_stores, generate_unified_vector_store
import evadb

def offline_setup():
    # 1) Connect to EvaDB
    cursor = evadb.connect("./evadb_data").cursor()  # use a consistent path
    print("Scraping job postings...")
    postings = scrape_palantir_jobs()
    print(f"Scraped {len(postings)} postings.")

    # 2) Create CSVs
    print("Creating CSVs from job postings...")
    doc_names = load_palantir_job_postings(postings)
    print(f"Created CSVs for {len(doc_names)} docs: {doc_names[:5]}...")

    # 3) Build vector store
    print("Initializing EvaDB vector store & indexes...")
    generate_unified_vector_store(cursor, doc_names)
    print("Done. Your data and indexes are now ready.")

    # 4) Optionally show tables
    df = cursor.query("SHOW TABLES;").df()
    print("Tables in EvaDB now:", df)

if __name__ == "__main__":
    offline_setup()