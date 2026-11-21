import requests
import pandas as pd
import json
import time
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# SETTINGS
# ============================================================

BASE = "https://data.bris.ac.uk/data"

SEARCH_API = (
    BASE
    + "/api/3/action/package_search"
)

OUT = Path(
    r"C:\Users\HP\Downloads\wildlive_api_discovery"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ROBUST SESSION
# ============================================================

session = requests.Session()

retry = Retry(
    total=6,
    connect=6,
    read=6,
    backoff_factor=2,
    status_forcelist=[
        429,
        500,
        502,
        503,
        504
    ],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(
    max_retries=retry
)

session.mount(
    "https://",
    adapter
)

session.headers.update({
    "User-Agent":
        "Mozilla/5.0 WildLive research downloader"
})


# ============================================================
# SEARCH CKAN
# ============================================================

def search_ckan(query):

    print(
        f"\nSearching CKAN for: {query}"
    )

    response = session.get(
        SEARCH_API,
        params={
            "q": query,
            "rows": 1000
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if not data.get(
        "success",
        False
    ):
        raise RuntimeError(
            "CKAN returned success=False"
        )

    return data[
        "result"
    ][
        "results"
    ]


# ============================================================
# QUERY FROM TWO DIRECTIONS
# ============================================================

packages = {}


for query in [
    "WildLive",
    "boxid"
]:

    try:

        results = search_ckan(
            query
        )

        print(
            "Packages returned:",
            len(results)
        )

        for package in results:

            package_id = package.get(
                "id"
            )

            if package_id:

                packages[
                    package_id
                ] = package

    except Exception as e:

        print(
            "\nSearch failed:",
            query
        )

        print(
            type(e).__name__,
            e
        )


print(
    "\nUnique packages collected:",
    len(packages)
)


# ============================================================
# FILTER WILDLIVE BOXID PACKAGES
# ============================================================

rows = []

raw_selected = []


for package in (
    packages.values()
):

    title = str(
        package.get(
            "title",
            ""
        )
    )

    name = str(
        package.get(
            "name",
            ""
        )
    )

    notes = str(
        package.get(
            "notes",
            ""
        )
    )


    search_text = (
        title
        + " "
        + name
        + " "
        + notes
    ).lower()


    # We only want WildLive box annotations

    if "boxid" not in search_text:
        continue


    # Most WildLive subpackages
    # should include WildLive in title/path.
    # Keep boxid packages anyway and inspect them.

    resources = package.get(
        "resources",
        []
    )


    json_resources = 0

    zip_resources = 0

    other_resources = 0


    resource_names = []


    for resource in resources:

        resource_name = str(
            resource.get(
                "name",
                ""
            )
        )

        resource_format = str(
            resource.get(
                "format",
                ""
            )
        ).lower()

        resource_url = str(
            resource.get(
                "url",
                ""
            )
        ).lower()


        resource_names.append(
            resource_name
        )


        if (
            resource_format == "json"
            or
            resource_url.endswith(
                ".json"
            )
            or
            resource_name.lower().endswith(
                ".json"
            )
        ):

            json_resources += 1


        elif (
            resource_format == "zip"
            or
            resource_url.endswith(
                ".zip"
            )
            or
            resource_name.lower().endswith(
                ".zip"
            )
        ):

            zip_resources += 1


        else:

            other_resources += 1


    rows.append({

        "package_title":
            title,

        "package_name":
            name,

        "package_id":
            package.get(
                "id",
                ""
            ),

        "package_url":

            BASE
            + "/dataset/"
            + str(
                package.get(
                    "id",
                    ""
                )
            ),

        "total_resources":
            len(resources),

        "json_resources":
            json_resources,

        "zip_resources":
            zip_resources,

        "other_resources":
            other_resources,

        "first_resource":
            (
                resource_names[0]
                if resource_names
                else ""
            ),

        "last_resource":
            (
                resource_names[-1]
                if resource_names
                else ""
            )
    })


    raw_selected.append(
        package
    )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    rows
)


if len(df) > 0:

    df = df.sort_values(
        [
            "package_title",
            "package_name"
        ]
    )


df.to_csv(

    OUT
    /
    "wildlive_boxid_discovery.csv",

    index=False
)


with open(

    OUT
    /
    "wildlive_boxid_packages_raw.json",

    "w",
    encoding="utf-8"

) as f:

    json.dump(
        raw_selected,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

with open(

    OUT
    /
    "SUMMARY.txt",

    "w",
    encoding="utf-8"

) as f:

    f.write(
        "WILDLIVE CKAN BOXID DISCOVERY\n"
    )

    f.write(
        "=" * 65
        + "\n\n"
    )

    f.write(
        f"Unique CKAN packages searched: "
        f"{len(packages)}\n"
    )

    f.write(
        f"Packages containing 'boxid': "
        f"{len(df)}\n\n"
    )


    if len(df) > 0:

        for _, row in (
            df.iterrows()
        ):

            f.write(
                f"{row['package_title']}\n"
            )

            f.write(
                f"ID: {row['package_id']}\n"
            )

            f.write(
                f"Resources: "
                f"{int(row['total_resources'])}\n"
            )

            f.write(
                f"JSON: "
                f"{int(row['json_resources'])}\n"
            )

            f.write(
                f"ZIP: "
                f"{int(row['zip_resources'])}\n\n"
            )


print(
    "\n===================================="
)

print(
    "WILDLIVE API DISCOVERY COMPLETE"
)

print(
    "===================================="
)


if len(df) == 0:

    print(
        "\nNo boxid packages were returned."
    )

    print(
        "Do NOT download anything yet."
    )

else:

    print(
        "\nBoxid packages found:"
    )

    print(
        df[
            [
                "package_title",
                "total_resources",
                "json_resources",
                "zip_resources"
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\nResults saved to:"
)

print(
    OUT
)