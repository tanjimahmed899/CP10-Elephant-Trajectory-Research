from pathlib import Path
import zipfile
import pandas as pd

ROOTS = [
    Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"),
    Path(r"C:\Users\HP\Downloads"),
]

OUT = Path(
    r"C:\Users\HP\Downloads\step6d_uncertainty_holdout"
)
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_TERMS = [
    "wildlive",
    "wildlife",
    "wild_box",
    "wildbox",
    "nestler",
    "elephant",
    "zebra",
    "giraffe",
    "tracklet",
    "trajectory",
]

USED_PATH_TERMS = [
    "final_neural_trajectory_dataset",
    "unified_trajectory_10hz",
    "wildbox_trajectory_dataset",
    "wildbox_fresh_confirmatory_results",
    "wildbox_fresh_holdout_zips",
    "baseline_constant_velocity",
    "baseline_kalman",
    "lstm_gru_results",
    "transformer_results",
    "transformer_ensemble_validation",
    "step2",
    "step3_fresh_wildbox_full_comparison",
    "step4",
    "step5",
    "step6_conformal_uncertainty",
    "step6b_scale_adaptive_conformal",
    "step6c_crossfit_conformal",
    "step6d_uncertainty_holdout",
]

ANNOTATION_EXTS = {
    ".csv", ".json", ".txt", ".xml", ".yaml", ".yml",
    ".mat", ".npy", ".npz", ".geojson"
}

MEDIA_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".m4v",
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

ARCHIVE_EXTS = {
    ".zip", ".7z", ".rar", ".tar", ".gz"
}


def contains_candidate_term(path):
    s = str(path).lower()
    return any(term in s for term in CANDIDATE_TERMS)


def is_known_used(path):
    s = str(path).lower()
    return any(term.lower() in s for term in USED_PATH_TERMS)


def classify(path):
    ext = path.suffix.lower()

    if ext in ANNOTATION_EXTS:
        return "annotation_like"

    if ext in MEDIA_EXTS:
        return "media"

    if ext in ARCHIVE_EXTS:
        return "archive"

    return "other"


def scan_files():
    rows = []

    for root in ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            if not contains_candidate_term(p):
                continue

            if is_known_used(p):
                used = True
            else:
                used = False

            rows.append({
                "path": str(p),
                "parent": str(p.parent),
                "name": p.name,
                "suffix": p.suffix.lower(),
                "category": classify(p),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
                "known_used_path": used,
                "contains_wildlive": "wildlive" in str(p).lower(),
                "contains_wildlife": "wildlife" in str(p).lower(),
                "contains_elephant": "elephant" in str(p).lower(),
                "contains_tracklet": "tracklet" in str(p).lower(),
                "contains_trajectory": "trajectory" in str(p).lower(),
            })

    return pd.DataFrame(rows)


def inspect_candidate_archives(df):
    rows = []

    for _, r in df.iterrows():
        if r["category"] != "archive":
            continue

        p = Path(r["path"])

        if p.suffix.lower() != ".zip":
            rows.append({
                "archive_path": str(p),
                "inspectable": False,
                "member_count": None,
                "annotation_like_members": None,
                "media_members": None,
                "wildlive_term_members": None,
                "note": "non-ZIP archive; not opened"
            })
            continue

        try:
            with zipfile.ZipFile(p, "r") as zf:
                members = [
                    x.filename
                    for x in zf.infolist()
                    if not x.is_dir()
                ]

            ann = 0
            media = 0
            wildlive_hits = 0

            for name in members:
                ext = Path(name).suffix.lower()

                if ext in ANNOTATION_EXTS:
                    ann += 1

                if ext in MEDIA_EXTS:
                    media += 1

                if "wildlive" in name.lower():
                    wildlive_hits += 1

            rows.append({
                "archive_path": str(p),
                "inspectable": True,
                "member_count": len(members),
                "annotation_like_members": ann,
                "media_members": media,
                "wildlive_term_members": wildlive_hits,
                "note": ""
            })

        except Exception as e:
            rows.append({
                "archive_path": str(p),
                "inspectable": False,
                "member_count": None,
                "annotation_like_members": None,
                "media_members": None,
                "wildlive_term_members": None,
                "note": f"{type(e).__name__}: {e}"
            })

    return pd.DataFrame(rows)


def build_parent_summary(df):
    if len(df) == 0:
        return pd.DataFrame()

    unused = df[df["known_used_path"] == False].copy()

    rows = []

    for parent, g in unused.groupby("parent"):
        rows.append({
            "parent": parent,
            "files": len(g),
            "annotation_like_files": int((g["category"] == "annotation_like").sum()),
            "media_files": int((g["category"] == "media").sum()),
            "archive_files": int((g["category"] == "archive").sum()),
            "total_size_mb": round(g["size_mb"].sum(), 3),
            "has_wildlive_term": bool(g["contains_wildlive"].any()),
            "has_elephant_term": bool(g["contains_elephant"].any()),
            "potential_ready_pair": bool(
                ((g["category"] == "annotation_like").sum() > 0)
                and
                ((g["category"] == "media").sum() > 0)
            ),
        })

    return pd.DataFrame(rows).sort_values(
        [
            "potential_ready_pair",
            "has_wildlive_term",
            "total_size_mb"
        ],
        ascending=[False, False, False]
    )


def main():
    files = scan_files()

    if len(files) == 0:
        files = pd.DataFrame(columns=[
            "path", "parent", "name", "suffix", "category",
            "size_mb", "known_used_path", "contains_wildlive",
            "contains_wildlife", "contains_elephant",
            "contains_tracklet", "contains_trajectory"
        ])

    files.to_csv(
        OUT / "step6d_local_candidate_files.csv",
        index=False
    )

    parent_summary = build_parent_summary(files)

    parent_summary.to_csv(
        OUT / "step6d_local_candidate_parent_summary.csv",
        index=False
    )

    archives = inspect_candidate_archives(
        files[files["known_used_path"] == False]
    )

    archives.to_csv(
        OUT / "step6d_local_candidate_archive_summary.csv",
        index=False
    )

    unused = files[
        files["known_used_path"] == False
    ].copy()

    with open(
        OUT / "STEP6D_F_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-F: LOCAL UNUSED DATASET CANDIDATE AUDIT\n"
        )
        f.write(
            "=" * 76 + "\n\n"
        )
        f.write(
            "No model predictions or forecast errors were computed.\n"
        )
        f.write(
            "This audit uses filenames, paths, extensions, and archive member counts only.\n\n"
        )

        f.write(
            f"Candidate files found: {len(files)}\n"
        )
        f.write(
            f"Unused-path candidate files: {len(unused)}\n\n"
        )

        f.write(
            "TOP UNUSED CANDIDATE PARENTS\n"
        )
        f.write(
            "----------------------------\n"
        )

        if len(parent_summary):
            f.write(
                parent_summary.head(50).to_string(index=False)
            )
        else:
            f.write("NONE")

        f.write(
            "\n\nUNUSED ARCHIVES\n"
        )
        f.write(
            "---------------\n"
        )

        if len(archives):
            f.write(
                archives.to_string(index=False)
            )
        else:
            f.write("NONE")

        f.write(
            "\n\nUNUSED WILDLIVE-NAMED FILES\n"
        )
        f.write(
            "---------------------------\n"
        )

        wildlive = unused[
            unused["contains_wildlive"] == True
        ]

        if len(wildlive):
            f.write(
                wildlive[
                    [
                        "path",
                        "category",
                        "size_mb"
                    ]
                ].to_string(index=False)
            )
        else:
            f.write("NONE")

    print(
        "\nSTEP 6D-F LOCAL UNUSED DATASET AUDIT COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )

    print(
        "\nUNUSED WILDLIVE-NAMED FILES:"
    )

    wildlive = unused[
        unused["contains_wildlive"] == True
    ]

    if len(wildlive):
        print(
            wildlive[
                [
                    "path",
                    "category",
                    "size_mb"
                ]
            ].to_string(index=False)
        )
    else:
        print("NONE")

    print(
        "\nTOP UNUSED CANDIDATE PARENTS:"
    )

    if len(parent_summary):
        print(
            parent_summary.head(30).to_string(index=False)
        )
    else:
        print("NONE")

    print(
        "\nUNUSED ARCHIVE SUMMARY:"
    )

    if len(archives):
        print(
            archives.to_string(index=False)
        )
    else:
        print("NONE")


if __name__ == "__main__":
    main()
