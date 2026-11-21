import json
from pathlib import Path

RAW = Path(
    r"C:\Users\HP\Downloads\wildlive_api_discovery"
    r"\wildlive_boxid_packages_raw.json"
)

with open(RAW, "r", encoding="utf-8") as f:
    packages = json.load(f)

TARGET = "WildLive/DJI_0207/boxid"

for package in packages:

    title = str(package.get("title", ""))

    if title == TARGET:

        print("\nPACKAGE")
        print("=" * 70)

        print("title:", package.get("title"))
        print("name :", package.get("name"))
        print("id   :", package.get("id"))

        resources = package.get("resources", [])

        print("\nTotal resources:", len(resources))

        print("\nFIRST 3 RESOURCES")
        print("=" * 70)

        for i, r in enumerate(resources[:3], start=1):

            print(f"\nRESOURCE {i}")
            print("-" * 50)

            for key in [
                "id",
                "name",
                "format",
                "mimetype",
                "url",
                "url_type",
                "cache_url",
                "package_id",
                "datastore_active"
            ]:

                print(
                    f"{key}:",
                    r.get(key)
                )

        break

else:

    print("Target package not found.")