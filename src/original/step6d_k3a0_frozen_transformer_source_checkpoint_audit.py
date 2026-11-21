from pathlib import Path
import ast
import json
import re
import hashlib
import pandas as pd

READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

BAMBI_LOCK = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k2b3_final_canonical"
    / "STEP6D_K2B3_CANONICAL_DATASET_LOCK.json"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_frozen_ensemble"
    / "source_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

MODEL_EXTS = {".pt", ".pth", ".ckpt", ".bin"}

HIGH_VALUE_FILES = [
    "evaluate_birdsai_testreal_frozen_ensemble.py",
    "evaluate_transformer_ensemble.py",
    "train_transformer.py",
    "step18b_evaluate_locked_geofence.py",
]

KEYWORDS = [
    "load_transformer",
    "transformer",
    "state_dict",
    "load_state_dict",
    "torch.load",
    "seed",
    "ensemble",
    "checkpoint",
    "model_path",
    "weights",
    "2s_to_2s",
    "2s_to_5s",
]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def safe_text(path):
    try:
        if path.stat().st_size > 5_000_000:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def find_function_sources(path, text):
    rows = []
    try:
        tree = ast.parse(text)
    except Exception:
        return rows

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        src = ast.get_source_segment(text, node) or ""
        low = src.lower()

        score = sum(1 for k in KEYWORDS if k.lower() in low)

        if score <= 0:
            continue

        rows.append({
            "file": str(path),
            "function": node.name,
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "score": int(score),
            "source": src,
        })

    return rows


def source_reference_rows(path, text):
    rows = []
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        low = line.lower()

        if not any(k.lower() in low for k in KEYWORDS):
            continue

        # Keep concise context around model/checkpoint/seed lines.
        lo = max(1, i - 2)
        hi = min(len(lines), i + 2)

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(lo, hi + 1)
        )

        rows.append({
            "file": str(path),
            "line": int(i),
            "text": line.strip(),
            "context": context,
        })

    return rows


def model_files_inventory():
    rows = []

    for path in READ_ONLY_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in MODEL_EXTS:
            continue

        try:
            size = int(path.stat().st_size)
            digest = sha256_file(path)
        except Exception:
            size = -1
            digest = "ERROR"

        name = path.name.lower()

        task_hint = None
        if "2s_to_2s" in name or "2to2" in name or "2s2s" in name:
            task_hint = "2s_to_2s"
        elif "2s_to_5s" in name or "2to5" in name or "2s5s" in name:
            task_hint = "2s_to_5s"

        seed_matches = re.findall(r"(?:seed|s)[_\-]?(\d+)", name)
        seed_hint = int(seed_matches[-1]) if seed_matches else None

        rows.append({
            "file": str(path),
            "name": path.name,
            "size_bytes": size,
            "sha256": digest,
            "task_hint": task_hint,
            "seed_hint": seed_hint,
        })

    return pd.DataFrame(rows)


def main():
    print("\nSTEP 6D-K3A0 — FROZEN TRANSFORMER ENSEMBLE SOURCE/CHECKPOINT AUDIT")
    print("Read-only source/model discovery only. No BAMBI predictions.\n")

    require(READ_ONLY_ROOT)
    require(BAMBI_LOCK)

    lock = json.loads(
        BAMBI_LOCK.read_text(
            encoding="utf-8"
        )
    )

    if lock.get("status") != "LOCKED_BEFORE_ANY_BAMBI_MODEL_PREDICTIONS":
        raise RuntimeError(
            "Unexpected BAMBI K2B3 lock status; refusing inference audit."
        )

    # Verify canonical files against frozen hashes.
    hash_checks = []

    for task, info in lock["task_outputs"].items():
        for field, hash_field in [
            ("canonical_npz", "canonical_npz_sha256"),
            ("canonical_metadata", "canonical_metadata_sha256"),
        ]:
            p = Path(info[field])
            require(p)

            actual = sha256_file(p)
            expected = info[hash_field]

            hash_checks.append({
                "task": task,
                "field": field,
                "path": str(p),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": bool(actual == expected),
            })

    hashes_df = pd.DataFrame(hash_checks)

    if not hashes_df["match"].all():
        raise RuntimeError(
            "Canonical dataset hash mismatch. Stop before model inference."
        )

    # Inventory checkpoint-like files.
    models_df = model_files_inventory()

    models_df.to_csv(
        OUT / "TABLE_step6dk3a0_model_checkpoint_inventory.csv",
        index=False
    )

    # Source audit.
    func_rows = []
    ref_rows = []

    py_files = list(READ_ONLY_ROOT.rglob("*.py"))

    for path in py_files:
        text = safe_text(path)
        if text is None:
            continue

        func_rows.extend(
            find_function_sources(path, text)
        )

        # Prioritize high-value files but still collect all.
        if path.name in HIGH_VALUE_FILES or any(
            k in text.lower()
            for k in ["load_transformer", "torch.load", "ensemble"]
        ):
            ref_rows.extend(
                source_reference_rows(path, text)
            )

    funcs_df = pd.DataFrame(func_rows)

    if len(funcs_df):
        funcs_df = funcs_df.sort_values(
            ["score", "file", "line_start"],
            ascending=[False, True, True],
        )

    refs_df = pd.DataFrame(ref_rows)

    funcs_df.to_csv(
        OUT / "TABLE_step6dk3a0_transformer_functions.csv",
        index=False
    )

    refs_df.to_csv(
        OUT / "step6dk3a0_source_reference_lines.csv",
        index=False
    )

    hashes_df.to_csv(
        OUT / "step6dk3a0_canonical_lock_hash_verification.csv",
        index=False
    )

    # Candidate loader functions.
    loader_candidates = pd.DataFrame()

    if len(funcs_df):
        mask = (
            funcs_df["function"]
            .str.lower()
            .str.contains(
                "load|transformer|ensemble|predict|evaluate",
                regex=True
            )
        )

        loader_candidates = funcs_df[mask].head(30).copy()

    # Candidate model files by naming/task clues.
    candidate_models = models_df.copy()

    if len(candidate_models):
        candidate_models["priority"] = 0

        candidate_models.loc[
            candidate_models["name"].str.lower().str.contains(
                "transformer"
            ),
            "priority",
        ] += 10

        candidate_models.loc[
            candidate_models["task_hint"].notna(),
            "priority",
        ] += 5

        candidate_models.loc[
            candidate_models["seed_hint"].notna(),
            "priority",
        ] += 3

        candidate_models = candidate_models.sort_values(
            ["priority", "file"],
            ascending=[False, True],
        )

    summary = {
        "step":
            "STEP_6D_K3A0_FROZEN_TRANSFORMER_SOURCE_CHECKPOINT_AUDIT",

        "canonical_lock_verified":
            bool(hashes_df["match"].all()),

        "checkpoint_files_found":
            int(len(models_df)),

        "transformer_candidate_functions":
            int(len(loader_candidates)),

        "status":
            (
                "FROZEN_MODEL_CANDIDATES_FOUND_PENDING_EXACT_TASK_SEED_MAPPING"
                if len(models_df) and len(loader_candidates)
                else
                "MODEL_MAPPING_INCOMPLETE_DO_NOT_RUN_BAMBI_INFERENCE"
            ),

        "important_rules": [
            "No BAMBI prediction is generated in K3A0.",
            "Canonical locked hashes are verified before any inference-related work.",
            "Exact frozen task/seed/checkpoint mapping must be source-confirmed; filenames alone are not sufficient.",
            "Do not retrain or recalibrate any model on BAMBI.",
        ],
    }

    (OUT / "STEP6D_K3A0_SUMMARY.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    with open(
        OUT / "STEP6D_K3A0_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K3A0 — FROZEN TRANSFORMER SOURCE/CHECKPOINT AUDIT\n"
        )
        f.write("=" * 82 + "\n\n")

        f.write("CANONICAL LOCK HASH VERIFICATION\n")
        f.write("--------------------------------\n")
        f.write(hashes_df.to_string(index=False))

        f.write("\n\nTOP CHECKPOINT CANDIDATES\n")
        f.write("-------------------------\n")
        f.write(
            candidate_models.head(40).to_string(index=False)
            if len(candidate_models)
            else "NONE"
        )

        f.write("\n\nTOP LOADER / MODEL FUNCTIONS\n")
        f.write("----------------------------\n")

        if len(loader_candidates):
            for _, r in loader_candidates.iterrows():
                f.write(
                    f"\nFILE: {r['file']}\n"
                    f"FUNCTION: {r['function']}\n"
                    f"LINES: {r['line_start']}-{r['line_end']}\n"
                    f"SCORE: {r['score']}\n"
                    f"{r['source']}\n"
                    + "-" * 80 + "\n"
                )
        else:
            f.write("NONE\n")

        f.write("\nSTATUS\n------\n")
        f.write(summary["status"] + "\n")

    print("CANONICAL LOCK HASH VERIFICATION:")
    print(hashes_df.to_string(index=False))

    print("\nTOP CHECKPOINT CANDIDATES:")
    print(
        candidate_models.head(40).to_string(index=False)
        if len(candidate_models)
        else "NONE"
    )

    print("\nTOP LOADER / MODEL FUNCTIONS:")
    if len(loader_candidates):
        print(
            loader_candidates[
                [
                    "file",
                    "function",
                    "line_start",
                    "line_end",
                    "score",
                ]
            ].to_string(index=False)
        )
    else:
        print("NONE")

    print("\nSTATUS:")
    print(summary["status"])

    print("\nSTEP 6D-K3A0 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
