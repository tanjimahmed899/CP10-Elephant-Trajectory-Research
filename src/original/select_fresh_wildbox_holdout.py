from pathlib import Path
import hashlib
import re

import pandas as pd
from huggingface_hub import HfApi


# ============================================================
# SETTINGS
# ============================================================

REPO_ID = "wildbox-anon-2026/wildbox-review"
REPO_TYPE = "dataset"

OUT = Path(
    r"C:\Users\HP\Downloads\wildbox_fresh_holdout_selection"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ALREADY INSPECTED — MUST NEVER ENTER FRESH HOLDOUT
# ============================================================

EXCLUDED_BASENAMES = {
    "DJI_20240119152552_0006_V.zip",
    "DJI_20240626143832_0004_V.zip",
    "DJI_20260227084907_0004_V.zip",
}


# ============================================================
# LIST REPOSITORY FILES ONLY
#
# IMPORTANT:
# No ZIP is downloaded.
# No frames/labels/tracks are inspected.
# ============================================================

api = HfApi()

print("Listing repository files only...")

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

print("Total repository files:", len(files))


# ============================================================
# FIND ELEPHANT VIDEO ZIP FILES
# ============================================================

rows = []

for path in files:

    basename = Path(path).name

    lower_path = path.lower()

    # Elephant folders / paths
    if "elep" not in lower_path:
        continue

    # Per-video ZIP
    if not basename.lower().endswith(".zip"):
        continue

    # Wildlife source video naming convention
    if not basename.startswith("DJI_"):
        continue

    # Detect elephant group from path:
    # elep1 / elep2 / elep3 / etc.
    match = re.search(
        r"(elep\d+)",
        lower_path
    )

    elephant_group = (
        match.group(1)
        if match
        else "unknown"
    )

    rows.append({
        "repo_path": path,
        "basename": basename,
        "elephant_group": elephant_group,
        "previously_used": basename in EXCLUDED_BASENAMES,
    })


candidates = pd.DataFrame(rows)

if len(candidates) == 0:

    raise RuntimeError(
        "No elephant DJI ZIP files were found. "
        "Do not guess paths. Share the output/error with me."
    )


candidates = candidates.sort_values(
    [
        "elephant_group",
        "basename"
    ]
).reset_index(
    drop=True
)


candidates.to_csv(
    OUT / "all_elephant_zip_candidates.csv",
    index=False
)


# ============================================================
# ELIGIBLE FRESH FILES
# ============================================================

eligible = candidates[
    ~candidates["previously_used"]
].copy()


if len(eligible) < 3:

    raise RuntimeError(
        f"Only {len(eligible)} unused elephant ZIPs found. "
        "Do not continue automatically."
    )


# ============================================================
# DETERMINISTIC SELECTION
#
# We intentionally DO NOT select by:
# - model performance
# - trajectory quality
# - number of elephants
# - duration
# - visual difficulty
#
# Hash depends only on repository path.
# ============================================================

def deterministic_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


eligible["selection_hash"] = (
    eligible["repo_path"]
    .apply(deterministic_hash)
)


selected_rows = []


# ============================================================
# PREFER ONE FRESH SOURCE FROM EACH ELEPHANT GROUP
# ============================================================

groups = sorted(
    g
    for g in eligible["elephant_group"].unique()
    if g != "unknown"
)


for group in groups:

    group_table = eligible[
        eligible["elephant_group"] == group
    ].copy()

    group_table = group_table.sort_values(
        "selection_hash"
    )

    if len(group_table) > 0:

        selected_rows.append(
            group_table.iloc[0]
        )


# ============================================================
# TARGET = 3 INDEPENDENT FRESH SOURCE VIDEOS
#
# If group-wise selection gives >3, hash-sort and retain 3.
# If <3, fill deterministically from remaining files.
# ============================================================

if len(selected_rows) > 0:

    selected = pd.DataFrame(
        selected_rows
    ).drop_duplicates(
        subset=["repo_path"]
    )

else:

    selected = pd.DataFrame(
        columns=eligible.columns
    )


if len(selected) > 3:

    selected = (
        selected.sort_values(
            "selection_hash"
        )
        .head(3)
        .copy()
    )


if len(selected) < 3:

    already_selected = set(
        selected["repo_path"].tolist()
    )

    remaining = eligible[
        ~eligible["repo_path"].isin(
            already_selected
        )
    ].sort_values(
        "selection_hash"
    )

    need = 3 - len(selected)

    selected = pd.concat(
        [
            selected,
            remaining.head(need)
        ],
        ignore_index=True
    )


selected = selected[
    [
        "repo_path",
        "basename",
        "elephant_group",
        "selection_hash"
    ]
].reset_index(
    drop=True
)


selected.insert(
    0,
    "holdout_id",
    [
        f"fresh_{i+1:02d}"
        for i in range(len(selected))
    ]
)


# ============================================================
# SAFETY CHECK
# ============================================================

assert len(selected) == 3

assert not any(
    selected["basename"].isin(
        EXCLUDED_BASENAMES
    )
)


# ============================================================
# SAVE LOCKED SELECTION
# ============================================================

selected.to_csv(
    OUT / "FRESH_HOLDOUT_LOCKED.csv",
    index=False
)


with open(
    OUT / "FRESH_HOLDOUT_PROTOCOL.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "WILDBOX FRESH EXTERNAL HOLDOUT — PREREGISTERED SELECTION\n"
    )

    f.write(
        "=" * 76 + "\n\n"
    )

    f.write(
        "PURPOSE\n"
    )

    f.write(
        "- Final confirmatory RGB external evaluation of the "
        "already-frozen Transformer ensemble.\n\n"
    )

    f.write(
        "SELECTION RULE\n"
    )

    f.write(
        "- Repository filenames were listed without downloading "
        "or inspecting new videos.\n"
    )

    f.write(
        "- Previously inspected WildBox source videos were excluded.\n"
    )

    f.write(
        "- Three unused elephant video ZIPs were selected "
        "deterministically using SHA-256 hashes of repository paths.\n"
    )

    f.write(
        "- Where possible, selection prefers one source from each "
        "elephant group.\n"
    )

    f.write(
        "- Selection did NOT use model performance, track quality, "
        "video content, trajectory statistics, or difficulty.\n\n"
    )

    f.write(
        "EXCLUDED PREVIOUSLY INSPECTED VIDEOS\n"
    )

    for name in sorted(EXCLUDED_BASENAMES):

        f.write(
            f"- {name}\n"
        )

    f.write(
        "\nLOCKED FRESH HOLDOUT\n"
    )

    for _, row in selected.iterrows():

        f.write(
            f"- {row['holdout_id']}: "
            f"{row['repo_path']}\n"
        )


print("\n========================================")
print("FRESH HOLDOUt SELECTION COMPLETE")
print("========================================")

print("\nSelected fresh source videos:\n")

print(
    selected[
        [
            "holdout_id",
            "elephant_group",
            "repo_path"
        ]
    ].to_string(
        index=False
    )
)

print("\nNO NEW VIDEO WAS DOWNLOADED.")
print("NO MODEL PERFORMANCE WAS EVALUATED.")

print("\nSaved to:")
print(OUT)