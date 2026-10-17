import pandas as pd
import re
from pathlib import Path
from unidecode import unidecode
from rapidfuzz.fuzz import ratio

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

MATCHING_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"

print("=" * 70)
print("FINAL SUBMISSION PIPELINE")
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
        "apartment": "apt",
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
# BLOCKING KEYS
# ============================================================

def name_keys(name, country):

    name = normalize_text(name)

    if not name:
        return []

    tokens = name.split()
    keys = set()

    # Exact normalized name
    keys.add(
        f"{country}|NAME|{name}"
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
            f"{country}|T2|{tokens[0][:4]}|{tokens[1][:4]}"
        )

    return keys


def address_keys(address, country):

    address = normalize_address(address)

    if not address:
        return []

    tokens = address.split()
    keys = set()

    # Exact address
    keys.add(
        f"{country}|ADDR|{address}"
    )

    # First + last address token
    if len(tokens) >= 2:

        keys.add(
            f"{country}|AF|{tokens[0][:5]}|{tokens[-1][:5]}"
        )

    return keys


# ============================================================
# LOAD TEST SOURCE 1
# ============================================================

print("\nLoading Test Source 1...")

s1 = pd.read_csv(
    TEST_SOURCE1,
    sep="\t"
)

print(
    "Source 1 rows:",
    len(s1)
)


# ============================================================
# BUILD SOURCE 1 BLOCKING INDEX
# ============================================================

print("\nCreating Source 1 blocking index...")

name_index = {}
address_index = {}

for row in s1.itertuples(index=False):

    country = str(row.country)

    for key in name_keys(
        row.business_name,
        country
    ):

        name_index.setdefault(
            key,
            set()
        ).add(row.entity_id)

    for key in address_keys(
        row.business_address,
        country
    ):

        address_index.setdefault(
            key,
            set()
        ).add(row.entity_id)


print(
    "Name blocking keys:",
    len(name_index)
)

print(
    "Address blocking keys:",
    len(address_index)
)


# ============================================================
# CANDIDATE STORAGE
# ============================================================

candidate_lookup = {
    entity_id: set()
    for entity_id in s1.entity_id
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
            "business_address",
            "country"
        ]
    ):

        total_rows += len(chunk)

        for row in chunk.itertuples(index=False):

            country = str(row.country)

            matched_s1 = set()

            # Name blocking
            for key in name_keys(
                row.business_name,
                country
            ):

                ids = name_index.get(key)

                if ids:
                    matched_s1.update(ids)

            # Address blocking
            for key in address_keys(
                row.business_address,
                country
            ):

                ids = address_index.get(key)

                if ids:
                    matched_s1.update(ids)

            for s1_id in matched_s1:

                if row.entity_id not in candidate_lookup[s1_id]:

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
    TEST_SOURCE2,
    "Source 2"
)

process_source(
    TEST_SOURCE3,
    "Source 3"
)


# ============================================================
# CREATE QUICK SOURCE LOOKUP FOR MATCHING
# ============================================================

print("\nBuilding candidate records...")

candidate_records = {}

for source_path in [
    TEST_SOURCE2,
    TEST_SOURCE3
]:

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

        for row in chunk.itertuples(index=False):

            candidate_records[row.entity_id] = (
                normalize_text(row.business_name),
                normalize_address(row.business_address),
                str(row.country)
            )


# ============================================================
# SOURCE 1 RECORD LOOKUP
# ============================================================

s1_records = {}

for row in s1.itertuples(index=False):

    s1_records[row.entity_id] = (
        normalize_text(row.business_name),
        normalize_address(row.business_address),
        str(row.country)
    )


# ============================================================
# MATCHING
# ============================================================

print("\nScoring candidates...")

matching_results = []
candidate_results = []

total_candidates = 0
total_matches = 0

for counter, s1_id in enumerate(
    s1.entity_id,
    start=1
):

    s1_name, s1_address, s1_country = s1_records[s1_id]

    candidates = candidate_lookup[s1_id]

    total_candidates += len(candidates)

    final_matches = []

    candidate_ids = []

    for candidate_id in candidates:

        candidate_ids.append(candidate_id)

        c_name, c_address, c_country = candidate_records[
            candidate_id
        ]

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        name_score = ratio(
            s1_name,
            c_name
        ) if s1_name and c_name else 0

        address_score = ratio(
            s1_address,
            c_address
        ) if s1_address and c_address else 0

        country_match = (
            s1_country == c_country
        )

        # Strong exact match
        exact_name = (
            s1_name != "" and
            s1_name == c_name
        )

        exact_address = (
            s1_address != "" and
            s1_address == c_address
        )

        # ----------------------------------------------------
        # PRECISION-HEAVY DECISION
        # ----------------------------------------------------

        accept = False

        if exact_name and country_match:
            accept = True

        elif (
            exact_name
            and address_score >= 70
        ):
            accept = True

        elif (
            exact_address
            and name_score >= 70
        ):
            accept = True

        elif (
            name_score >= 92
            and address_score >= 75
            and country_match
        ):
            accept = True

        if accept:
            final_matches.append(
                candidate_id
            )

    # Remove duplicates
    final_matches = list(
        dict.fromkeys(final_matches)
    )

    candidate_ids = list(
        dict.fromkeys(candidate_ids)
    )

    total_matches += len(final_matches)

    matching_results.append(
        (
            s1_id,
            ",".join(final_matches)
        )
    )

    candidate_results.append(
        (
            s1_id,
            ",".join(candidate_ids)
        )
    )

    if counter % 100000 == 0:

        print(
            f"Matched {counter:,} / "
            f"{len(s1):,} Source 1 records"
        )


# ============================================================
# WRITE MATCHING RESULTS
# ============================================================

print("\nWriting matching_results.tsv...")

matching_df = pd.DataFrame(
    matching_results,
    columns=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

matching_df.to_csv(
    MATCHING_OUTPUT,
    sep="\t",
    index=False
)


# ============================================================
# WRITE CANDIDATE PAIRS
# ============================================================

print("Writing candidate_pairs.tsv...")

candidate_df = pd.DataFrame(
    candidate_results,
    columns=[
        "source1_entity_id",
        "candidate_entity_ids"
    ]
)

candidate_df.to_csv(
    CANDIDATE_OUTPUT,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PIPELINE COMPLETE")
print("=" * 70)

print(
    "Source 1 records:",
    len(s1)
)

print(
    "Total candidate pairs:",
    f"{total_candidates:,}"
)

print(
    "Total predicted matches:",
    f"{total_matches:,}"
)

print(
    "\nMatching file:",
    MATCHING_OUTPUT
)

print(
    "Candidate file:",
    CANDIDATE_OUTPUT
)

print("\nNext step:")
print(
    "Run utils/validate_submission.py"
)

print("=" * 70)