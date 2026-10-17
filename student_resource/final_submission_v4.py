import os
import re
from collections import defaultdict

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio


# ============================================================
# FAST ENTITY RESOLUTION V4
# MULTI-BLOCK TRAINING VALIDATION
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")

TRAIN = os.path.join(DATASET, "train")
TEST = os.path.join(DATASET, "test")

S1_FILE = os.path.join(TRAIN, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN, "train_ground_truth.tsv")


# ============================================================
# NORMALIZATION
# ============================================================

def norm(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace("&", " and ")

    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def compact(value):
    return norm(value).replace(" ", "")


# ============================================================
# SIMILARITY
# ============================================================

def similarity(s1, s2):

    n1 = norm(s1)
    n2 = norm(s2)

    if not n1 or not n2:
        return 0.0

    if n1 == n2:
        return 100.0

    r1 = ratio(n1, n2)
    r2 = token_set_ratio(n1, n2)

    return max(r1, r2)


def match_score(
    s1_name,
    s1_addr,
    s2_name,
    s2_addr,
    s1_country,
    s2_country
):

    if s1_country != s2_country:
        return -1

    name_score = similarity(
        s1_name,
        s2_name
    )

    addr_score = similarity(
        s1_addr,
        s2_addr
    )

    exact_name = (
        norm(s1_name) == norm(s2_name)
        and norm(s1_name) != ""
    )

    exact_addr = (
        norm(s1_addr) == norm(s2_addr)
        and norm(s1_addr) != ""
    )

    # --------------------------------------------------------
    # VERY STRONG MATCHES
    # --------------------------------------------------------

    if exact_name and exact_addr:
        return 100.0

    if exact_name and addr_score >= 65:
        return 96.0

    if exact_addr and name_score >= 65:
        return 95.0

    # --------------------------------------------------------
    # FUZZY COMBINED SCORE
    # --------------------------------------------------------

    score = (
        0.60 * name_score +
        0.40 * addr_score
    )

    if name_score >= 94 and addr_score >= 60:
        score += 4

    if name_score >= 90 and addr_score >= 75:
        score += 5

    return min(score, 100.0)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 70)
print("ENTITY RESOLUTION V4 - FAST MULTI-BLOCK VALIDATION")
print("=" * 70)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
).fillna("")

print(
    "Source 1 rows:",
    len(s1)
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt_df = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

ground_truth = {}

for row in gt_df.itertuples(index=False):

    s1_id = row.source1_entity_id

    value = str(
        row.matched_entity_ids
    ).strip()

    if value:

        ground_truth[s1_id] = set(
            x.strip()
            for x in value.split(",")
            if x.strip()
        )

    else:

        ground_truth[s1_id] = set()


print(
    "Ground truth rows:",
    len(ground_truth)
)


# ============================================================
# INDEXES
# ============================================================

name_index = defaultdict(set)

name_token_index = defaultdict(set)

compact_name_index = defaultdict(set)

address_index = defaultdict(set)

address_token_index = defaultdict(set)

address_number_index = defaultdict(set)

source_records = {}


# ============================================================
# SOURCE LOADER
# ============================================================

def load_source(path, label):

    print("\nLoading", label)

    total = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    for chunk in reader:

        chunk = chunk.fillna("")

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id

            business_name = row.business_name
            business_address = row.business_address
            country = row.country

            source_records[entity_id] = (
                business_name,
                business_address,
                country
            )

            # ==================================================
            # NAME INDEXES
            # ==================================================

            n = norm(
                business_name
            )

            if n:

                name_index[
                    (country, n)
                ].add(entity_id)

                # Compact name
                c = compact(
                    business_name
                )

                if c and c != n:

                    compact_name_index[
                        (country, c)
                    ].add(entity_id)

                # Longest substantial token
                tokens = n.split()

                if tokens:

                    longest = max(
                        tokens,
                        key=len
                    )

                    if len(longest) >= 5:

                        name_token_index[
                            (country, longest)
                        ].add(entity_id)

            # ==================================================
            # ADDRESS INDEXES
            # ==================================================

            a = norm(
                business_address
            )

            if a:

                address_index[
                    (country, a)
                ].add(entity_id)

                # Longest address token
                address_tokens = a.split()

                if address_tokens:

                    longest_address = max(
                        address_tokens,
                        key=len
                    )

                    if len(longest_address) >= 6:

                        address_token_index[
                            (
                                country,
                                longest_address
                            )
                        ].add(entity_id)

                # Numeric components
                numbers = re.findall(
                    r"\d{4,}",
                    a
                )

                for number in numbers:

                    address_number_index[
                        (country, number)
                    ].add(entity_id)

        total += len(chunk)

        if total % 1000000 == 0:

            print(
                f"  loaded {total:,} rows"
            )

    print(
        label,
        "loaded:",
        f"{total:,} rows"
    )


# ============================================================
# LOAD SOURCE 2 + SOURCE 3
# ============================================================

load_source(
    S2_FILE,
    "Source 2"
)

load_source(
    S3_FILE,
    "Source 3"
)

print(
    "\nSource records indexed:",
    f"{len(source_records):,}"
)


# ============================================================
# MATCH TRAINING SOURCE 1
# ============================================================

print(
    "\nMatching training Source 1..."
)

predictions = {}

processed = 0


for row in s1.itertuples(
    index=False
):

    s1_id = row.entity_id

    country = row.country

    name = norm(
        row.business_name
    )

    address = norm(
        row.business_address
    )

    candidates = set()


    # ========================================================
    # BLOCK 1
    # EXACT NORMALIZED NAME
    # ========================================================

    if name:

        candidates.update(
            name_index.get(
                (
                    country,
                    name
                ),
                ()
            )
        )


    # ========================================================
    # BLOCK 2
    # COMPACT NAME
    # ========================================================

    if name:

        compact_value = (
            name.replace(" ", "")
        )

        candidates.update(
            compact_name_index.get(
                (
                    country,
                    compact_value
                ),
                ()
            )
        )


    # ========================================================
    # BLOCK 3
    # LONGEST NAME TOKEN
    # ========================================================

    name_tokens = name.split()

    if name_tokens:

        longest_name_token = max(
            name_tokens,
            key=len
        )

        if len(longest_name_token) >= 5:

            candidates.update(
                name_token_index.get(
                    (
                        country,
                        longest_name_token
                    ),
                    ()
                )
            )


    # ========================================================
    # BLOCK 4
    # EXACT ADDRESS
    # ========================================================

    if address:

        candidates.update(
            address_index.get(
                (
                    country,
                    address
                ),
                ()
            )
        )


    # ========================================================
# BLOCK 5
# LONGEST ADDRESS TOKEN
# ========================================================

address_tokens = address.split()

if address_tokens:

    longest_address_token = max(
        address_tokens,
        key=len
    )

    if len(longest_address_token) >= 6:

        candidates.update(
            address_token_index.get(
                (
                    country,
                    longest_address_token
                ),
                ()
            )
        )


    # ========================================================
    # BLOCK 6
    # ADDRESS NUMBER / PIN
    # ========================================================

    numbers = re.findall(
        r"\d{4,}",
        address
    )

    for number in numbers:

        candidates.update(
            address_number_index.get(
                (
                    country,
                    number
                ),
                ()
            )
        )


    # ========================================================
    # SCORE CANDIDATES
    # ========================================================

    accepted = []

    for cid in candidates:

        c_name, c_addr, c_country = (
            source_records[cid]
        )

        score = match_score(
            row.business_name,
            row.business_address,
            c_name,
            c_addr,
            country,
            c_country
        )


        # ----------------------------------------------------
        # HIGH CONFIDENCE
        # ----------------------------------------------------

        if score >= 96:

            accepted.append(
                (
                    cid,
                    score
                )
            )


        # ----------------------------------------------------
        # STRONG FUZZY MATCH
        # ----------------------------------------------------

        elif score >= 92:

            ns = similarity(
                row.business_name,
                c_name
            )

            ads = similarity(
                row.business_address,
                c_addr
            )

            if (
                ns >= 94
                and
                ads >= 65
            ):

                accepted.append(
                    (
                        cid,
                        score
                    )
                )


    # ========================================================
    # SORT
    # ========================================================

    accepted.sort(
        key=lambda x: x[1],
        reverse=True
    )


    predictions[s1_id] = {
        x[0]
        for x in accepted
    }


    processed += 1


    if processed % 10000 == 0:

        print(
            f"Processed "
            f"{processed:,}/"
            f"{len(s1):,}"
        )


# ============================================================
# F0.5 EVALUATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING VALIDATION"
)

print(
    "=" * 70
)


def f05(
    precision,
    recall
):

    if (
        precision == 0
        and
        recall == 0
    ):

        return 0.0

    return (
        1.25
        *
        precision
        *
        recall
        /
        (
            0.25
            *
            precision
            +
            recall
        )
    )


total_tp = 0

total_fp = 0

total_fn = 0

entity_scores = []


for s1_id in ground_truth:

    truth = ground_truth[
        s1_id
    ]

    pred = predictions.get(
        s1_id,
        set()
    )


    tp = len(
        pred & truth
    )

    fp = len(
        pred - truth
    )

    fn = len(
        truth - pred
    )


    total_tp += tp

    total_fp += fp

    total_fn += fn


    precision = (

        tp / len(pred)

        if pred

        else 0.0
    )


    recall = (

        tp / len(truth)

        if truth

        else (
            1.0
            if not pred
            else 0.0
        )
    )


    entity_scores.append(
        f05(
            precision,
            recall
        )
    )


# ============================================================
# FINAL METRICS
# ============================================================

macro_f05 = (

    sum(entity_scores)
    /
    len(entity_scores)

    if entity_scores

    else 0
)


precision = (

    total_tp
    /
    (
        total_tp
        +
        total_fp
    )

    if (
        total_tp
        +
        total_fp
    )

    else 0
)


recall = (

    total_tp
    /
    (
        total_tp
        +
        total_fn
    )

    if (
        total_tp
        +
        total_fn
    )

    else 0
)


micro_f05 = f05(
    precision,
    recall
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nRESULTS")

print(
    "-" * 50
)

print(
    "True positives:",
    f"{total_tp:,}"
)

print(
    "False positives:",
    f"{total_fp:,}"
)

print(
    "False negatives:",
    f"{total_fn:,}"
)

print(
    "Precision:",
    f"{precision:.6f}"
)

print(
    "Recall:",
    f"{recall:.6f}"
)

print(
    "Micro F_0.5:",
    f"{micro_f05:.6f}"
)

print(
    "Macro F_0.5:",
    f"{macro_f05:.6f}"
)


print(
    "\n" + "=" * 70
)

print(
    "V4 TRAINING TEST COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nV3 baseline Macro F_0.5: 0.408872"
)

print(
    f"V4 Macro F_0.5: {macro_f05:.6f}"
)

if macro_f05 > 0.408872:

    print(
        "\nV4 IMPROVED OVER V3"
    )

else:

    print(
        "\nV4 DID NOT IMPROVE OVER V3"
    )

print(
    "\nDo NOT generate the final test submission yet."
)