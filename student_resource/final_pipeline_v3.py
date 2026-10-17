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
print("ENTITY RESOLUTION - BLOCKING V3")
print("=" * 70)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = unidecode(str(text)).lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

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
        "apartment": "apt"
    }

    for old, new in replacements.items():

        text = re.sub(
            r"\b" + re.escape(old) + r"\b",
            new,
            text
        )

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# RESTRICTIVE BLOCKING KEYS
# ============================================================

def make_block_keys(name, address, country):

    name = normalize_text(name)
    address = normalize_address(address)

    keys = set()

    # --------------------------------------------------------
    # NAME KEYS
    # --------------------------------------------------------

    if name:

        tokens = name.split()

        # Exact normalized name
        keys.add(
            f"{country}|NAME|{name}"
        )

        # First 4 compact characters
        compact = "".join(tokens)

        if len(compact) >= 4:
            keys.add(
                f"{country}|P4|{compact[:4]}"
            )

        # First + last token
        if len(tokens) >= 2:

            first = tokens[0]
            last = tokens[-1]

            if len(first) >= 3 and len(last) >= 3:

                keys.add(
                    f"{country}|FL|{first[:4]}|{last[:4]}"
                )

        # First two tokens
        if len(tokens) >= 2:

            keys.add(
                f"{country}|T2|"
                f"{tokens[0][:4]}|"
                f"{tokens[1][:4]}"
            )

    # --------------------------------------------------------
    # ADDRESS KEYS
    # --------------------------------------------------------

    if address:

        address_tokens = address.split()

        # Exact normalized address
        keys.add(
            f"{country}|ADDR|{address}"
        )

        # First 8 compact address characters
        compact_address = "".join(address_tokens)

        if len(compact_address) >= 8:

            keys.add(
                f"{country}|A8|{compact_address[:8]}"
            )

        # First address token + last address token
        if len(address_tokens) >= 2:

            keys.add(
                f"{country}|AF|"
                f"{address_tokens[0][:5]}|"
                f"{address_tokens[-1][:5]}"
            )

    return keys


# ============================================================
# LOAD SOURCE 1
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
# GROUND TRUTH
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
# CREATE SOURCE 1 BLOCKING INDEX
# ============================================================

print("\nCreating Source 1 blocking index...")

source1_block_lookup = {}

for row in validation_s1.itertuples(index=False):

    keys = make_block_keys(
        row.business_name,
        row.business_address,
        row.country
    )

    for key in keys:

        source1_block_lookup.setdefault(
            key,
            set()
        ).add(row.entity_id)


print(
    "Unique blocking keys:",
    len(source1_block_lookup)
)


# ============================================================
# REMOVE VERY COMMON BLOCKING KEYS
# ============================================================

# A very common key creates thousands of false candidates.
# We keep exact NAME and exact ADDRESS keys even when common.
# Other broad keys are limited.

MAX_BROAD_KEY_SIZE = 50

filtered_block_lookup = {}

for key, ids in source1_block_lookup.items():

    if (
        key.startswith("|NAME|")
        or key.startswith("|ADDR|")
    ):
        filtered_block_lookup[key] = ids

    elif len(ids) <= MAX_BROAD_KEY_SIZE:
        filtered_block_lookup[key] = ids


source1_block_lookup = filtered_block_lookup

print(
    "Usable blocking keys:",
    len(source1_block_lookup)
)


# ============================================================
# CANDIDATE STORAGE
# ============================================================

candidate_lookup = {
    entity_id: set()
    for entity_id in validation_s1["entity_id"]
}


# ============================================================
# PROCESS SOURCE
# ============================================================

def process_source(source_path, source_name):

    print("\n" + "-" * 60)
    print("Processing", source_name)
    print("-" * 60)

    total_rows = 0
    raw_pairs = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        chunksize=100000,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ):

        total_rows += len(chunk)

        for row in chunk.itertuples(index=False):

            keys = make_block_keys(
                row.business_name,
                row.business_address,
                row.country
            )

            matched_s1 = set()

            for key in keys:

                ids = source1_block_lookup.get(
                    key
                )

                if ids:
                    matched_s1.update(ids)

            for s1_id in matched_s1:

                if row.entity_id not in candidate_lookup[s1_id]:

                    candidate_lookup[s1_id].add(
                        row.entity_id
                    )

                    raw_pairs += 1

        if total_rows % 1000000 == 0:

            print(
                f"Processed {total_rows:,} rows | "
                f"Unique candidate pairs: {raw_pairs:,}"
            )

    print(
        source_name,
        "rows processed:",
        f"{total_rows:,}"
    )

    print(
        source_name,
        "unique candidate pairs:",
        f"{raw_pairs:,}"
    )


# ============================================================
# RUN BLOCKING
# ============================================================

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
print("BLOCKING V3 EVALUATION")
print("=" * 70)

total_true = 0
found_true = 0

records_all_found = 0
records_missing = 0

for row in validation_s1.itertuples(index=False):

    true_string = row.true_matches

    if not true_string.strip():
        continue

    true_ids = set(
        x.strip()
        for x in true_string.split(",")
        if x.strip()
    )

    candidates = candidate_lookup[
        row.entity_id
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


average_candidates = (
    sum(
        len(x)
        for x in candidate_lookup.values()
    )
    / len(candidate_lookup)
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

print(
    f"Average candidates per Source 1: "
    f"{average_candidates:.2f}"
)

print("\n" + "=" * 70)
print("V3 BLOCKING TEST COMPLETE")
print("=" * 70)