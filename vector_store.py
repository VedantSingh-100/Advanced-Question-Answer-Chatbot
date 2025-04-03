import os
import evadb
import tqdm
import time

def generate_vector_stores(cursor, docs):
    """
    For each doc in docs:
      1) Drop table if it exists.
      2) Create a table named `doc` with the appropriate schema.
      3) LOAD CSV <file_path> INTO `doc`.
      4) Create the SentenceFeatureExtractor function (if not already created).
      5) Create a `doc_features` table to hold the extracted embeddings + metadata.
      6) Create an index for vector searches on the features column.

    Finally, print the columns of the first doc's features table for verification.
    """
    evadb_path = os.path.dirname(evadb.__file__)
    cursor.query(
        f"""
        CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
        IMPL '{evadb_path}/functions/sentence_feature_extractor.py';
        """
    ).df()

    # 2) For each doc, drop table if exists, create table, load CSV, create features table, index it.
    for doc in tqdm.tqdm(docs, desc="Indexing docs"):
        start_time = time.time()
        # Drop any previous table named `doc`
        cursor.query(f"DROP TABLE IF EXISTS {doc};").df()

        # Create a fresh table for the CSV schema
        cursor.query(f"""
        CREATE TABLE {doc} (
        job_id TEXT,
        job_title TEXT,
        commitment TEXT,
        department TEXT,
        team TEXT,
        level TEXT,
        location TEXT,
        all_locations TEXT,
        country TEXT,
        workplace_type TEXT,
        tags TEXT,
        description TEXT,
        bullet_sections TEXT,
        closing_text TEXT,
        chunk_id INTEGER,
        data TEXT);""").df()


        # Construct the file path for the current doc
        file_path = f"data/palantir_careers/{doc}.csv"

        # Load the CSV into the new table
        load_result = cursor.query(f"LOAD CSV '{file_path}' INTO {doc};").df()
        cursor.query(f"DROP TABLE IF EXISTS {doc}_features;").df()  # <-- Force re-creation

        cursor.query(
            f"""
            CREATE TABLE {doc}_features AS
            SELECT
                SentenceFeatureExtractor(data),
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
            FROM {doc};
            """
        ).df()
        # Create an index on the features column for vector searches
        cursor.query(
            f"CREATE INDEX IF NOT EXISTS {doc}_index ON {doc}_features (features) USING FAISS;"
        ).df()

        elapsed = time.time() - start_time
        print(f"✅ Finished {doc} in {elapsed:.2f} seconds.\n")

def generate_unified_vector_store(cursor, docs):
    """
    Create one global table (all_jobs), load every CSV into it,
    then build a single features table (all_jobs_features) with a single FAISS index.
    This version omits "AS features" to avoid EVA binding errors.
    """
    evadb_path = os.path.dirname(evadb.__file__)

    # 1) Create the feature extraction function if not already present
    cursor.query(f"""
        CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
        IMPL '{evadb_path}/functions/sentence_feature_extractor.py';
    """).df()

    # 2) Drop the "all_jobs" table if it already exists
    cursor.query("DROP TABLE IF EXISTS all_jobs;").df()

    # 3) Create a single table for *all* job postings
    cursor.query(f"""
        CREATE TABLE all_jobs (
            doc_name TEXT,
            job_id TEXT,
            job_title TEXT,
            commitment TEXT,
            department TEXT,
            team TEXT,
            level TEXT,
            location TEXT,
            all_locations TEXT,
            country TEXT,
            workplace_type TEXT,
            tags TEXT,
            description TEXT,
            bullet_sections TEXT,
            closing_text TEXT,
            chunk_id INTEGER,
            data TEXT
        );
    """).df()

    # 4) For each CSV doc, load into a TEMP table, then insert into all_jobs with doc_name
    for doc in tqdm.tqdm(docs, desc="Loading docs into 'all_jobs'"):
        start_time = time.time()
        file_path = f"data/palantir_careers/{doc}.csv"
        cursor.query(f"LOAD CSV '{file_path}' INTO all_jobs;").df()

    # 5) Create the "all_jobs_features" table by extracting embeddings
    #    Notice we do NOT use "AS features" here
    cursor.query("DROP TABLE IF EXISTS all_jobs_features;").df()
    cursor.query(f"""
        CREATE TABLE all_jobs_features AS
        SELECT
            SentenceFeatureExtractor(data),   -- no "AS features"
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
        FROM all_jobs;
    """).df()

    # 6) Create a single FAISS index on the function call directly
    #    This references the auto-generated column name, e.g. "SentenceFeatureExtractor(data)"
    cursor.query("""
        CREATE INDEX IF NOT EXISTS all_jobs_index
        ON all_jobs_features (SentenceFeatureExtractor(data))
        USING FAISS;
    """).df()

    print("\n✅ Finished creating unified vector store (all_jobs_features + single FAISS index).")

def table_exists(cursor, table_name: str) -> bool:
    df = cursor.query("SHOW TABLES;").df()  # no LIKE here!
    # 'name' is typically the column with the table names.
    if "name" not in df.columns:
        # Unexpected schema; maybe EvaDB changed. Return False or handle gracefully.
        return False

    table_list = df["name"].tolist()
    return table_name in table_list

