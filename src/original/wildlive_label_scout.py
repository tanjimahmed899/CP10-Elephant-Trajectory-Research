import json
import math
import time
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# PATHS
# ============================================================

RAW_FILE = Path(
    r"C:\Users\HP\Downloads\wildlive_api_discovery"
    r"\wildlive_boxid_packages_raw.json"
)

OUT = Path(
    r"C:\Users\HP\Downloads\wildlive_label_scout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Evenly sample up to 80 annotation files per sequence.
# This is only a scouting stage.
SAMPLES_PER_SEQUENCE = 20

# Keep this moderate so Bristol server is not hammered.
MAX_WORKERS = 4

TIMEOUT = 8


# ============================================================
# SESSION
# ============================================================

def make_session():

    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],
        allowed_methods=["GET"]
    )

    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=retry
        )
    )

    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 WildLive research annotation scout"
    })

    return session


# One session per worker call is safest enough here.
def fetch_json(url):

    session = make_session()

    try:

        r = session.get(
            url,
            timeout=TIMEOUT
        )

        r.raise_for_status()

        return r.json(), None

    except Exception as e:

        return None, str(e)


# ============================================================
# LOAD DISCOVERY
# ============================================================

with open(
    RAW_FILE,
    "r",
    encoding="utf-8"
) as f:

    packages = json.load(f)


# ============================================================
# FIND MAIN BOXID PACKAGES
# ============================================================

sequence_packages = []


for package in packages:

    title = str(
        package.get(
            "title",
            ""
        )
    )

    # Ignore nested __boxid packages.
    if "/__boxid" in title:
        continue

    if not title.lower().endswith(
        "/boxid"
    ):
        continue


    resources = package.get(
        "resources",
        []
    )


    json_resources = []


    for r in resources:

        name = str(
            r.get(
                "name",
                ""
            )
        )

        fmt = str(
            r.get(
                "format",
                ""
            )
        ).lower()

        url = str(
            r.get(
                "url",
                ""
            )
        )


        if (

            fmt == "json"

            or

            name.lower().endswith(
                ".json"
            )

            or

            url.lower().endswith(
                ".json"
            )

        ):

            if url:

                json_resources.append(
                    r
                )


    if len(json_resources) == 0:
        continue


    # Extract sequence name:
    # WildLive/SEQUENCE/boxid

    parts = title.split("/")

    sequence = (
        parts[-2]
        if len(parts) >= 2
        else title
    )


    sequence_packages.append({

        "sequence":
            sequence,

        "title":
            title,

        "resources":
            json_resources
    })


print(
    "\nSequences with JSON boxid annotations:",
    len(sequence_packages)
)


# ============================================================
# EVEN SAMPLING
# ============================================================

def evenly_sample(
    resources,
    n
):

    total = len(
        resources
    )

    if total <= n:

        return resources


    indices = [

        round(
            i
            *
            (
                total - 1
            )
            /
            (
                n - 1
            )
        )

        for i in range(n)

    ]


    # Remove accidental duplicates

    indices = sorted(
        set(indices)
    )


    return [

        resources[i]

        for i in indices

    ]


# ============================================================
# PARSE LABELME-LIKE JSON
# ============================================================

def extract_shapes(data):

    if isinstance(
        data,
        dict
    ):

        if (
            "shapes"
            in data
            and
            isinstance(
                data["shapes"],
                list
            )
        ):

            return data[
                "shapes"
            ]


        # fallback

        for value in (
            data.values()
        ):

            if (

                isinstance(
                    value,
                    list
                )

                and

                len(value) > 0

                and

                isinstance(
                    value[0],
                    dict
                )

            ):

                # likely shapes
                if any(

                    (
                        "label"
                        in item
                    )

                    for item
                    in value
                    if isinstance(
                        item,
                        dict
                    )

                ):

                    return value


    elif isinstance(
        data,
        list
    ):

        return data


    return []


# ============================================================
# BUILD DOWNLOAD JOBS
# ============================================================

jobs = []


for package in (
    sequence_packages
):

    sequence = package[
        "sequence"
    ]

    resources = package[
        "resources"
    ]


    sampled = evenly_sample(

        resources,

        SAMPLES_PER_SEQUENCE

    )


    print(

        f"{sequence}: "
        f"{len(resources)} total JSON, "
        f"{len(sampled)} sampled"

    )


    for resource in sampled:

        jobs.append({

            "sequence":
                sequence,

            "resource_name":
                resource.get(
                    "name",
                    ""
                ),

            "url":
                resource.get(
                    "url",
                    ""
                )
        })


# ============================================================
# DOWNLOAD + AUDIT
# ============================================================

sequence_labels = defaultdict(
    Counter
)

sequence_tracks = defaultdict(
    lambda: defaultdict(set)
)

sequence_success = Counter()

sequence_fail = Counter()

detailed_rows = []


with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:


    future_map = {

        executor.submit(
            fetch_json,
            job["url"]
        ):
        job

        for job in jobs

    }


    completed = 0


    for future in as_completed(
        future_map
    ):

        job = future_map[
            future
        ]

        sequence = job[
            "sequence"
        ]


        data, error = (
            future.result()
        )


        completed += 1


        if error is not None:

            sequence_fail[
                sequence
            ] += 1

            continue


        sequence_success[
            sequence
        ] += 1


        shapes = extract_shapes(
            data
        )


        labels_in_file = []


        for shape in shapes:

            if not isinstance(
                shape,
                dict
            ):
                continue


            label = str(

                shape.get(
                    "label",
                    "UNKNOWN"
                )

            ).strip()


            group_id = shape.get(
                "group_id",
                None
            )


            sequence_labels[
                sequence
            ][
                label
            ] += 1


            labels_in_file.append(
                label
            )


            if group_id is not None:

                sequence_tracks[
                    sequence
                ][
                    label
                ].add(
                    str(
                        group_id
                    )
                )


        detailed_rows.append({

            "sequence":
                sequence,

            "resource_name":
                job[
                    "resource_name"
                ],

            "num_shapes":
                len(
                    shapes
                ),

            "labels":
                "|".join(
                    sorted(
                        set(
                            labels_in_file
                        )
                    )
                )
        })


        if completed % 100 == 0:

            print(
                f"Processed "
                f"{completed}/{len(jobs)}"
            )


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for package in (
    sequence_packages
):

    sequence = package[
        "sequence"
    ]


    labels = sequence_labels[
        sequence
    ]


    all_labels = sorted(
        labels.keys()
    )


    elephant_labels = [

        label

        for label
        in all_labels

        if "elephant"
        in label.lower()

    ]


    elephant_boxes = sum(

        labels[label]

        for label
        in elephant_labels

    )


    elephant_track_ids = set()


    for label in elephant_labels:

        elephant_track_ids.update(

            sequence_tracks[
                sequence
            ][
                label
            ]

        )


    summary_rows.append({

        "sequence":
            sequence,

        "total_json_resources":
            len(
                package[
                    "resources"
                ]
            ),

        "sampled_success":
            sequence_success[
                sequence
            ],

        "sampled_failed":
            sequence_fail[
                sequence
            ],

        "labels_seen":
            " | ".join(
                all_labels
            ),

        "elephant_seen":
            int(
                len(
                    elephant_labels
                )
                >
                0
            ),

        "sampled_elephant_boxes":
            elephant_boxes,

        "sampled_unique_elephant_ids":
            len(
                elephant_track_ids
            )
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df = summary_df.sort_values(

    [
        "elephant_seen",
        "sampled_unique_elephant_ids",
        "sampled_elephant_boxes"
    ],

    ascending=[
        False,
        False,
        False
    ]

)


summary_df.to_csv(

    OUT /
    "sequence_label_scout.csv",

    index=False
)


pd.DataFrame(
    detailed_rows
).to_csv(

    OUT /
    "sampled_json_details.csv",

    index=False
)


# ============================================================
# TEXT SUMMARY
# ============================================================

with open(

    OUT /
    "SUMMARY.txt",

    "w",
    encoding="utf-8"

) as f:


    f.write(
        "WILDLIVE LABEL SCOUT\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )


    f.write(

        f"Sequences audited: "
        f"{len(sequence_packages)}\n"

    )


    f.write(

        f"Sample target per sequence: "
        f"{SAMPLES_PER_SEQUENCE}\n\n"

    )


    for _, row in (
        summary_df.iterrows()
    ):


        f.write(

            f"{row['sequence']}\n"

        )


        f.write(

            f"JSON resources: "
            f"{int(row['total_json_resources'])}\n"

        )


        f.write(

            f"Successful samples: "
            f"{int(row['sampled_success'])}\n"

        )


        f.write(

            f"Failed samples: "
            f"{int(row['sampled_failed'])}\n"

        )


        f.write(

            f"Labels: "
            f"{row['labels_seen']}\n"

        )


        f.write(

            f"Elephant seen: "
            f"{int(row['elephant_seen'])}\n"

        )


        f.write(

            f"Sample elephant boxes: "
            f"{int(row['sampled_elephant_boxes'])}\n"

        )


        f.write(

            f"Sample elephant IDs: "
            f"{int(row['sampled_unique_elephant_ids'])}\n\n"

        )


print(
    "\n===================================="
)

print(
    "WILDLIVE LABEL SCOUT COMPLETE"
)

print(
    "====================================\n"
)


print(

    summary_df[
        [
            "sequence",
            "labels_seen",
            "elephant_seen",
            "sampled_elephant_boxes",
            "sampled_unique_elephant_ids"
        ]
    ].to_string(
        index=False
    )

)


print(
    "\nResults saved:"
)

print(
    OUT
)