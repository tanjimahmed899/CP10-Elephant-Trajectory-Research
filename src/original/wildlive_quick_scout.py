import json
from pathlib import Path
from collections import Counter, defaultdict

import requests
import pandas as pd


RAW_FILE = Path(
    r"C:\Users\HP\Downloads\wildlive_api_discovery"
    r"\wildlive_boxid_packages_raw.json"
)

OUT = Path(
    r"C:\Users\HP\Downloads\wildlive_quick_scout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLES_PER_SEQUENCE = 3


with open(
    RAW_FILE,
    "r",
    encoding="utf-8"
) as f:

    packages = json.load(f)


def extract_shapes(data):

    if isinstance(data, dict):

        if (
            "shapes" in data
            and isinstance(
                data["shapes"],
                list
            )
        ):
            return data["shapes"]

    if isinstance(data, list):
        return data

    return []


def evenly_sample(
    resources,
    n
):

    if len(resources) <= n:
        return resources

    indices = [
        0,
        len(resources) // 2,
        len(resources) - 1
    ]

    return [
        resources[i]
        for i in indices
    ]


rows = []


for package in packages:

    title = str(
        package.get(
            "title",
            ""
        )
    )

    if "/__boxid" in title:
        continue

    if not title.lower().endswith(
        "/boxid"
    ):
        continue


    resources = []


    for resource in package.get(
        "resources",
        []
    ):

        name = str(
            resource.get(
                "name",
                ""
            )
        )

        fmt = str(
            resource.get(
                "format",
                ""
            )
        ).lower()

        url = str(
            resource.get(
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

            resources.append(
                resource
            )


    if not resources:
        continue


    sequence = title.split(
        "/"
    )[-2]


    sampled = evenly_sample(
        resources,
        SAMPLES_PER_SEQUENCE
    )


    print(
        f"\n=== {sequence} ==="
    )


    labels = Counter()

    elephant_ids = set()

    success = 0

    failed = 0


    for i, resource in enumerate(
        sampled,
        start=1
    ):

        url = resource.get(
            "url",
            ""
        )


        print(
            f"Request {i}/{len(sampled)}...",
            end=" ",
            flush=True
        )


        try:

            response = requests.get(

                url,

                timeout=5,

                headers={
                    "User-Agent":
                    "Mozilla/5.0"
                }
            )


            response.raise_for_status()

            data = response.json()

            shapes = extract_shapes(
                data
            )


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


                labels[
                    label
                ] += 1


                if (
                    "elephant"
                    in label.lower()
                ):

                    group_id = shape.get(
                        "group_id"
                    )


                    if group_id is not None:

                        elephant_ids.add(
                            str(
                                group_id
                            )
                        )


            success += 1

            print(
                "OK"
            )


        except Exception as e:

            failed += 1

            print(
                "FAILED"
            )


    elephant_boxes = sum(

        count

        for label, count
        in labels.items()

        if "elephant"
        in label.lower()

    )


    rows.append({

        "sequence":
            sequence,

        "total_json":
            len(resources),

        "successful_samples":
            success,

        "failed_samples":
            failed,

        "labels_seen":
            " | ".join(
                sorted(
                    labels.keys()
                )
            ),

        "elephant_seen":
            int(
                elephant_boxes > 0
            ),

        "elephant_boxes_sampled":
            elephant_boxes,

        "elephant_ids_sampled":
            len(
                elephant_ids
            )
    })


df = pd.DataFrame(
    rows
)


df.to_csv(

    OUT /
    "quick_scout.csv",

    index=False
)


with open(

    OUT /
    "SUMMARY.txt",

    "w",
    encoding="utf-8"

) as f:


    f.write(
        "WILDLIVE QUICK LABEL SCOUT\n"
    )

    f.write(
        "=" * 60
        +
        "\n\n"
    )


    for _, row in (
        df.iterrows()
    ):


        f.write(

            f"{row['sequence']}\n"

        )


        f.write(

            f"Success: "
            f"{row['successful_samples']}\n"

        )


        f.write(

            f"Failed: "
            f"{row['failed_samples']}\n"

        )


        f.write(

            f"Labels: "
            f"{row['labels_seen']}\n"

        )


        f.write(

            f"Elephant seen: "
            f"{row['elephant_seen']}\n"

        )


        f.write(

            f"Elephant boxes sampled: "
            f"{row['elephant_boxes_sampled']}\n"

        )


        f.write(

            f"Elephant IDs sampled: "
            f"{row['elephant_ids_sampled']}\n\n"

        )


print(
    "\nQUICK SCOUT COMPLETE"
)

print(
    df.to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUT
)