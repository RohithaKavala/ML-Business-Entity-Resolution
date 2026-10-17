import pandas as pd
import re
from pathlib import Path
from unidecode import unidecode

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

print("=" * 70)
print("FINAL ENTITY RESOLUTION PIPELINE")
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
# BLOCKING KEY
# ============================================================

def make_block_keys(name, country):

    name = normalize_text(name)

    if not name:
        return []

    tokens = name.split()

    keys = set()

    # Country + first 4 characters of normalized name
    compact = "".join(tokens)

    if len(compact) >= 4:
        keys.add(
            f"{country}|P4|{compact[:4]}"
        )

    # Country + first token
    if tokens:

        first_token = tokens[0]

        if len(first_token) >= 3:

            keys.add(
                f"{country}|T1|{first_token[:6]}"
            )

    # Country + first two tokens
    if len(tokens) >= 2:

        keys.add(
            f"{country}|T2|{tokens[0][:4]}|{tokens[1][:4]}"
        )

    return list(keys)


# ============================================================
# LOAD SOURCE 1 VALIDATION SAMPLE
# ============================================================

print("\nLoading Source 1...")

source1 = pd.read_csv(
    TRAIN_SOURCE1,
    sep="\t"
)

validation_s1 = source1.sample(
    n=10000,
    random_state=42
).copy()

print(
    "Validation Source 1 records:",
    len(validation_s1)
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

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


# ============================================================
# CREATE BLOCKING INDEX FROM SOURCE 1
# ============================================================

print("\nCreating Source 1 blocking keys...")

source1_block_lookup = {}

for _, row in validation_s1.iterrows():

    keys = make_block_keys(
        row["business_name"],
        row["country"]
    )

    for key in keys:

        source1_block_lookup.setdefault(
            key,
            []
        ).append(
            row["entity_id"]
        )

print(
    "Unique blocking keys:",
    len(source1_block_lookup)
)


# ============================================================
# CANDIDATE LOOKUP
# ============================================================

candidate_lookup = {
    entity_id: set()
    for entity_id in validation_s1["entity_id"]
}


# ============================================================
# PROCESS SOURCE 2 / SOURCE 3
# ============================================================

def process_source(source_path, source_name):

    print("\n" + "-" * 60)
    print("Processing", source_name)
    print("-" * 60)

    total_rows = 0
    candidate_pairs = 0

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

        total_rows += len(chunk)

        for row in chunk.itertuples(index=False):

            keys = make_block_keys(
                row.business_name,
                row.country
            )

            matched_s1 = set()

            for key in keys:

                ids = source1_block_lookup.get(
                    key,
                    []
                )

                matched_s1.update(ids)

            for s1_id in matched_s1:

                candidate_lookup[s1_id].add(
                    row.entity_id
                )

                candidate_pairs += 1

        if total_rows % 1000000 == 0:

            print(
                f"Processed {total_rows:,} rows | "
                f"Candidate pairs: {candidate_pairs:,}"
            )

    print(
        source_name,
        "rows processed:",
        f"{total_rows:,}"
    )

    print(
        source_name,
        "candidate pairs:",
        f"{candidate_pairs:,}"
    )


process_source(
    TRAIN_SOURCE2,
    "Source 2"
)

process_source(
    TRAIN_SOURCE3,
    "Source 3"
)


# ============================================================
# EVALUATE BLOCKING
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING EVALUATION")
print("=" * 70)

total_true = 0
found_true = 0

records_all_found = 0
records_missing = 0

for _, row in validation_s1.iterrows():

    true_string = row["true_matches"]

    if not true_string.strip():
        continue

    true_ids = set(
        x.strip()
        for x in true_string.split(",")
        if x.strip()
    )

    candidates = candidate_lookup[
        row["entity_id"]
    ]

    found = true_ids.intersection(
        candidates
    )

    total_true += len(true_ids)
    found_true += len(found)

    if true_ids.issubset(candidates):

        records_all_found += 1

    else:

        records_missing += 1


recall = (
    found_true / total_true
    if total_true
    else 0
)

print(
    "\nTrue match IDs:",
    total_true
)

print(
    "True matches found:",
    found_true
)

print(
    f"Blocking recall: {recall * 100:.2f}%"
)

print(
    "Records with ALL true matches:",
    records_all_found
)

print(
    "Records missing at least one match:",
    records_missing
)

average_candidates = (
    sum(
        len(x)
        for x in candidate_lookup.values()
    )
    / len(candidate_lookup)
)

print(
    f"Average candidates per Source 1: "
    f"{average_candidates:.2f}"
)

print("\nPipeline blocking test complete.")