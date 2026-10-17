import pandas as pd
import re
import sqlite3
import csv
import shutil
from pathlib import Path
from unidecode import unidecode

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "dataset" / "test"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"

DB_FILE = BASE_DIR / "submission_index.db"

CHUNK_SIZE = 100000

print("=" * 70)
print("OPTIMIZED FINAL SUBMISSION PIPELINE V2")
print("=" * 70)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = unidecode(str(text)).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def normalize_address(text):

    if pd.isna(text):
        return ""

    text = unidecode(str(text)).lower()

    replacements = {
        "street": "st",
        "road": "rd",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "highway": "hwy",
        "apartment": "apt",
        "private limited": "pvt ltd",
        "limited": "ltd",
        "corporation": "corp",
        "incorporated": "inc"
    }

    for old, new in replacements.items():

        text = re.sub(
            r"\b" + re.escape(old) + r"\b",
            new,
            text
        )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# CREATE SQLITE DATABASE
# ============================================================

print("\nPreparing local candidate index...")

if DB_FILE.exists():

    print("Existing index found:")
    print(DB_FILE)

    print("Reusing existing index.")

else:

    print("Creating new SQLite index...")
    print("This is a one-time operation.")

    conn = sqlite3.connect(DB_FILE)

    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-100000")

    conn.execute("""
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT,
            business_address TEXT,
            country TEXT,
            name_norm TEXT,
            address_norm TEXT
        )
    """)

    # --------------------------------------------------------
    # SOURCE 2 + SOURCE 3
    # --------------------------------------------------------

    def load_source(path, source_name):

        print("\nLoading", source_name)

        total = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=CHUNK_SIZE,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ):

            chunk["name_norm"] = (
                chunk["business_name"]
                .map(normalize_text)
            )

            chunk["address_norm"] = (
                chunk["business_address"]
                .map(normalize_address)
            )

            rows = []

            for row in chunk.itertuples(index=False):

                rows.append((
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    row.country,
                    row.name_norm,
                    row.address_norm
                ))

            conn.executemany(
                """
                INSERT OR IGNORE INTO records
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows
            )

            conn.commit()

            total += len(chunk)

            if total % 1000000 == 0:

                print(
                    f"{source_name}: "
                    f"{total:,} rows indexed"
                )

        print(
            f"{source_name} complete: "
            f"{total:,} rows"
        )

    load_source(
        TEST_SOURCE2,
        "Source 2"
    )

    load_source(
        TEST_SOURCE3,
        "Source 3"
    )

    print("\nCreating indexes...")

    conn.execute("""
        CREATE INDEX idx_name_country
        ON records(name_norm, country)
    """)

    conn.execute("""
        CREATE INDEX idx_address_country
        ON records(address_norm, country)
    """)

    conn.execute("""
        CREATE INDEX idx_name
        ON records(name_norm)
    """)

    conn.execute("""
        CREATE INDEX idx_address
        ON records(address_norm)
    """)

    conn.commit()

    print("Indexes created.")

    conn.close()


# ============================================================
# OPEN DATABASE
# ============================================================

conn = sqlite3.connect(
    DB_FILE
)

cursor = conn.cursor()


# ============================================================
# TEST SOURCE 1
# ============================================================

print("\n" + "-" * 60)
print("Loading Test Source 1")
print("-" * 60)

source1 = pd.read_csv(
    TEST_SOURCE1,
    sep="\t"
)

print(
    "Source 1 rows:",
    f"{len(source1):,}"
)


# ============================================================
# OUTPUT FILES
# ============================================================

candidate_out = open(
    CANDIDATE_FILE,
    "w",
    encoding="utf-8",
    newline=""
)

matching_out = open(
    MATCHING_FILE,
    "w",
    encoding="utf-8",
    newline=""
)

candidate_writer = csv.writer(
    candidate_out,
    delimiter="\t"
)

matching_writer = csv.writer(
    matching_out,
    delimiter="\t"
)

candidate_writer.writerow([
    "source1_entity_id",
    "candidate_entity_ids"
])

matching_writer.writerow([
    "source1_entity_id",
    "matched_entity_ids"
])


# ============================================================
# PROCESS TEST SOURCE 1
# ============================================================

print("\nGenerating candidates + matches...")

total_rows = len(source1)

matched_rows = 0
candidate_rows = 0
empty_rows = 0

for index, row in enumerate(
    source1.itertuples(index=False),
    start=1
):

    s1_id = row.entity_id

    name_norm = normalize_text(
        row.business_name
    )

    address_norm = normalize_address(
        row.business_address
    )

    country = "" if pd.isna(row.country) else str(row.country)

    candidate_ids = set()

    # --------------------------------------------------------
    # BLOCK 1: EXACT NORMALIZED NAME + COUNTRY
    # --------------------------------------------------------

    if name_norm:

        cursor.execute(
            """
            SELECT entity_id
            FROM records
            WHERE name_norm = ?
              AND country = ?
            """,
            (
                name_norm,
                country
            )
        )

        for result in cursor.fetchall():

            candidate_ids.add(
                result[0]
            )

    # --------------------------------------------------------
    # BLOCK 2: EXACT NORMALIZED ADDRESS + COUNTRY
    # --------------------------------------------------------

    if address_norm:

        cursor.execute(
            """
            SELECT entity_id
            FROM records
            WHERE address_norm = ?
              AND country = ?
            """,
            (
                address_norm,
                country
            )
        )

        for result in cursor.fetchall():

            candidate_ids.add(
                result[0]
            )

    # --------------------------------------------------------
    # WRITE CANDIDATES
    # --------------------------------------------------------

    candidate_list = sorted(
        candidate_ids
    )

    candidate_writer.writerow([
        s1_id,
        ",".join(candidate_list)
    ])

    candidate_rows += len(
        candidate_list
    )

    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    matched_ids = []

    if candidate_list:

        # First preference:
        # exact name + exact address

        if name_norm and address_norm:

            cursor.execute(
                """
                SELECT entity_id
                FROM records
                WHERE name_norm = ?
                  AND address_norm = ?
                  AND country = ?
                """,
                (
                    name_norm,
                    address_norm,
                    country
                )
            )

            matched_ids = [
                x[0]
                for x in cursor.fetchall()
            ]

        # Second preference:
        # exact name only
        #
        # Only use this when there is no
        # exact name+address match.

        if not matched_ids and name_norm:

            cursor.execute(
                """
                SELECT entity_id
                FROM records
                WHERE name_norm = ?
                  AND country = ?
                """,
                (
                    name_norm,
                    country
                )
            )

            name_matches = [
                x[0]
                for x in cursor.fetchall()
            ]

            # Precision protection:
            # only accept exact-name matching when
            # there are a small number of candidates.

            if len(name_matches) <= 5:

                matched_ids = name_matches

    matching_writer.writerow([
        s1_id,
        ",".join(
            sorted(set(matched_ids))
        )
    ])

    if matched_ids:
        matched_rows += 1
    else:
        empty_rows += 1

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 10000 == 0:

        candidate_out.flush()
        matching_out.flush()

        print(
            f"Processed "
            f"{index:,}/{total_rows:,} "
            f"Source 1 rows | "
            f"Matched: {matched_rows:,} | "
            f"Empty: {empty_rows:,} | "
            f"Candidates: {candidate_rows:,}"
        )


# ============================================================
# CLOSE
# ============================================================

candidate_out.close()
matching_out.close()

conn.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PIPELINE COMPLETE")
print("=" * 70)

print(
    "Source 1 rows:",
    f"{total_rows:,}"
)

print(
    "Rows with matches:",
    f"{matched_rows:,}"
)

print(
    "Rows with no matches:",
    f"{empty_rows:,}"
)

print(
    "Total candidate IDs:",
    f"{candidate_rows:,}"
)

print("\nOutput files:")

print(
    "Candidate:",
    CANDIDATE_FILE
)

print(
    "Matching:",
    MATCHING_FILE
)

print("\nIMPORTANT:")
print("Run validate_submission.py before submitting.")