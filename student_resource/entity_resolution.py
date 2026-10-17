import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Dataset paths
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

print("Entity Resolution project started successfully!")
print("Project folder:", BASE_DIR)
print("\nLoading Source 1 training data...")

train_s1 = pd.read_csv(
    TRAIN_SOURCE1,
    sep="\t"
)

print("Source 1 loaded successfully!")
print("Rows:", len(train_s1))
print("Columns:", train_s1.columns.tolist())
print("\nCreating validation sample...")

# Use 10,000 Source 1 records for a quick local validation
validation_s1 = train_s1.sample(
    n=10000,
    random_state=42
)

print("Validation records:", len(validation_s1))
print(validation_s1.head())
import re
from unidecode import unidecode

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = unidecode(str(text)).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


validation_s1["name_normalized"] = (
    validation_s1["business_name"]
    .apply(normalize_text)
)

print("\nNormalized names:")
print(
    validation_s1[
        ["business_name", "name_normalized"]
    ].head(10).to_string(index=False)
)
def normalize_address(text):
    if pd.isna(text):
        return ""

    text = unidecode(str(text)).lower()

    # Common address abbreviations
    replacements = {
        "street": "st",
        "road": "rd",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "highway": "hwy",
        "apartment": "apt"
    }

    for old, new in replacements.items():
        text = re.sub(r"\b" + old + r"\b", new, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


validation_s1["address_normalized"] = (
    validation_s1["business_address"]
    .apply(normalize_address)
)

print("\nNormalized addresses:")
print(
    validation_s1[
        ["business_address", "address_normalized"]
    ].head(10).to_string(index=False)
)
print("\nLoading ground truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH,
    sep="\t"
).fillna("")

truth_lookup = dict(
    zip(
        ground_truth["source1_entity_id"],
        ground_truth["matched_entity_ids"]
    )
)

validation_s1["true_matches"] = (
    validation_s1["entity_id"]
    .map(truth_lookup)
    .fillna("")
)

print(
    "Validation records with matches:",
    (validation_s1["true_matches"].str.strip() != "").sum()
)

print(
    "Validation records with no matches:",
    (validation_s1["true_matches"].str.strip() == "").sum()
)
print("\nTesting chunked Source 2 reading...")

source2_rows = 0

for chunk in pd.read_csv(
    TRAIN_SOURCE2,
    sep="\t",
    chunksize=100000
):
    source2_rows += len(chunk)
    print("First chunk loaded:", len(chunk), "rows")
    break

print("Chunked reading works!")
print("\nFinding Source 2 candidates...")

# Normalize the validation names once
validation_names = set(validation_s1["name_normalized"])

candidate_count = 0

for chunk in pd.read_csv(
    TRAIN_SOURCE2,
    sep="\t",
    chunksize=100000
):
    chunk["name_normalized"] = (
        chunk["business_name"].apply(normalize_text)
    )

    # Keep only rows whose normalized name occurs
    # in our 10,000 validation records
    candidates = chunk[
        chunk["name_normalized"].isin(validation_names)
    ]

    candidate_count += len(candidates)

print("Total Source 2 candidates:", candidate_count)
print("\n" + "=" * 60)
print("TESTING CANDIDATE GENERATION FOR SOURCE 2 + SOURCE 3")
print("=" * 60)

# Candidate lookup:
# normalized business name -> Source 1 validation IDs
validation_name_lookup = {}

for _, row in validation_s1.iterrows():
    name = row["name_normalized"]

    if name:
        validation_name_lookup.setdefault(name, []).append(
            row["entity_id"]
        )


def find_candidates(source_path, source_name):
    print(f"\nProcessing {source_name}...")

    total_rows = 0
    total_candidates = 0
    chunks_processed = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        chunksize=100000,
        usecols=[
            "entity_id",
            "business_name",
            "country"
        ]
    ):
        chunks_processed += 1
        total_rows += len(chunk)

        chunk["name_normalized"] = (
            chunk["business_name"]
            .apply(normalize_text)
        )

        candidates = chunk[
            chunk["name_normalized"].isin(
                validation_name_lookup.keys()
            )
        ]

        total_candidates += len(candidates)

        if chunks_processed % 10 == 0:
            print(
                f"Processed {total_rows:,} rows | "
                f"Candidates found: {total_candidates:,}"
            )

    print(f"{source_name} rows processed: {total_rows:,}")
    print(f"{source_name} candidates: {total_candidates:,}")

    return total_candidates


source2_candidate_count = find_candidates(
    TRAIN_SOURCE2,
    "Source 2"
)

source3_candidate_count = find_candidates(
    TRAIN_SOURCE3,
    "Source 3"
)

print("\n" + "=" * 60)
print("CANDIDATE GENERATION SUMMARY")
print("=" * 60)

print(
    "Source 2 candidates:",
    source2_candidate_count
)

print(
    "Source 3 candidates:",
    source3_candidate_count
)

print(
    "Total candidates:",
    source2_candidate_count + source3_candidate_count
)
print("\n" + "=" * 60)
print("CHECKING BLOCKING RECALL")
print("=" * 60)

# Collect candidate IDs for our validation Source 1 records
candidate_lookup = {
    entity_id: set()
    for entity_id in validation_s1["entity_id"]
}

# Check Source 2
for chunk in pd.read_csv(
    TRAIN_SOURCE2,
    sep="\t",
    chunksize=100000,
    usecols=["entity_id", "business_name"]
):
    chunk["name_normalized"] = (
        chunk["business_name"].apply(normalize_text)
    )

    candidates = chunk[
        chunk["name_normalized"].isin(validation_name_lookup.keys())
    ]

    for _, row in candidates.iterrows():
        for s1_id in validation_name_lookup[row["name_normalized"]]:
            candidate_lookup[s1_id].add(row["entity_id"])


# Check Source 3
for chunk in pd.read_csv(
    TRAIN_SOURCE3,
    sep="\t",
    chunksize=100000,
    usecols=["entity_id", "business_name"]
):
    chunk["name_normalized"] = (
        chunk["business_name"].apply(normalize_text)
    )

    candidates = chunk[
        chunk["name_normalized"].isin(validation_name_lookup.keys())
    ]

    for _, row in candidates.iterrows():
        for s1_id in validation_name_lookup[row["name_normalized"]]:
            candidate_lookup[s1_id].add(row["entity_id"])


# Calculate whether every true match was included
total_true_matches = 0
found_true_matches = 0
records_with_all_true_matches = 0
records_with_missing_true_matches = 0

for _, row in validation_s1.iterrows():

    s1_id = row["entity_id"]
    true_string = row["true_matches"]

    if not true_string.strip():
        continue

    true_ids = set(
        x.strip()
        for x in true_string.split(",")
        if x.strip()
    )

    candidates = candidate_lookup.get(s1_id, set())

    found = true_ids.intersection(candidates)

    total_true_matches += len(true_ids)
    found_true_matches += len(found)

    if true_ids.issubset(candidates):
        records_with_all_true_matches += 1
    else:
        records_with_missing_true_matches += 1


if total_true_matches > 0:
    blocking_recall = (
        found_true_matches / total_true_matches
    )
else:
    blocking_recall = 0


print("\nBlocking evaluation:")
print("True match IDs:", total_true_matches)
print("True matches found in candidates:", found_true_matches)
print(
    "Blocking recall:",
    round(blocking_recall * 100, 2),
    "%"
)
print(
    "Records with ALL true matches:",
    records_with_all_true_matches
)
print(
    "Records missing at least one true match:",
    records_with_missing_true_matches
)
print("\n" + "=" * 60)
print("TESTING ADDRESS-BASED BLOCKING")
print("=" * 60)

# Create lookup using normalized address
validation_address_lookup = {}

for _, row in validation_s1.iterrows():
    address = row["address_normalized"]

    if address:
        validation_address_lookup.setdefault(
            address, []
        ).append(row["entity_id"])


def address_blocking_recall(source_path, source_name):

    print(f"\nProcessing {source_name} by address...")

    candidate_lookup_address = {
        entity_id: set()
        for entity_id in validation_s1["entity_id"]
    }

    total_candidates = 0
    total_rows = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        chunksize=100000,
        usecols=[
            "entity_id",
            "business_address"
        ]
    ):

        total_rows += len(chunk)

        chunk["address_normalized"] = (
            chunk["business_address"]
            .apply(normalize_address)
        )

        candidates = chunk[
            chunk["address_normalized"].isin(
                validation_address_lookup.keys()
            )
        ]

        total_candidates += len(candidates)

        for _, row in candidates.iterrows():

            address = row["address_normalized"]

            for s1_id in validation_address_lookup[address]:

                candidate_lookup_address[s1_id].add(
                    row["entity_id"]
                )

    # Evaluate recall
    true_total = 0
    found_total = 0

    for _, row in validation_s1.iterrows():

        true_string = row["true_matches"]

        if not true_string.strip():
            continue

        true_ids = set(
            x.strip()
            for x in true_string.split(",")
            if x.strip()
        )

        candidates = candidate_lookup_address[
            row["entity_id"]
        ]

        true_total += len(true_ids)

        found_total += len(
            true_ids.intersection(candidates)
        )

    recall = (
        found_total / true_total
        if true_total > 0
        else 0
    )

    print(f"{source_name} rows processed: {total_rows:,}")
    print(f"{source_name} address candidates: {total_candidates:,}")
    print(f"True matches found: {found_total:,}")
    print(f"Address blocking recall: {recall * 100:.2f}%")

    return candidate_lookup_address


address_candidates_s2 = address_blocking_recall(
    TRAIN_SOURCE2,
    "Source 2"
)

address_candidates_s3 = address_blocking_recall(
    TRAIN_SOURCE3,
    "Source 3"
)