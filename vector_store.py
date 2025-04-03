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
    for doc in tqdm(docs, desc="Indexing docs"):
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

def table_exists(cursor, table_name: str) -> bool:
    df = cursor.query("SHOW TABLES;").df()  # no LIKE here!
    # 'name' is typically the column with the table names.
    if "name" not in df.columns:
        # Unexpected schema; maybe EvaDB changed. Return False or handle gracefully.
        return False

    table_list = df["name"].tolist()
    return table_name in table_list

