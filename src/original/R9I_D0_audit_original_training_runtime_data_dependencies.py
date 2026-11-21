
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

TRANSFORMER_SRC = MAIN1 / "train_transformer.py"
RECURRENT_SRC = MAIN1 / "train_lstm_gru.py"

R9I_B_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_B_exact_model_hyperparameter_and_split_lock"
    / "R9I_B_MANIFEST.json"
)

R9I_C_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_C_V2_matched_isotropic_trainreal_dataset"
    / "R9I_C_V2_MANIFEST.json"
)

ISO_DATASET_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_C_V2_matched_isotropic_trainreal_dataset"
    / "dataset"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_D0_training_runtime_dependency_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    TRANSFORMER_SRC,
    RECURRENT_SRC,
]

TARGET_FUNCTIONS = {
    "train_transformer.py": [
        "train_one",
        "evaluate",
    ],
    "train_lstm_gru.py": [
        "train_one",
        "evaluate_dataset",
    ],
}

DATASET_TOKENS = [
    "birdsai_trainval_canonical",
    "birdsai_trainval_metadata",
    "birdsai_test_canonical",
    "birdsai_test_metadata",
    "wildbox_external_canonical",
    "wildbox_external_metadata",
]

OUTPUT_TOKENS = [
    "transformer_results",
    "lstm_gru_results",
    "predictions",
    ".pt",
    ".csv",
    ".json",
    ".txt",
]


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_locks():
    for p in [
        R9I_B_MANIFEST,
        R9I_C_MANIFEST,
        ISO_DATASET_DIR,
        TRANSFORMER_SRC,
        RECURRENT_SRC,
    ]:
        require(p)

    b = json.loads(
        R9I_B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    c = json.loads(
        R9I_C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        b.get("status")
        !=
        "R9I_B_EXACT_MODEL_HYPERPARAMETER_SPLIT_AND_CHECKPOINT_LOCK_COMPLETE_READY_FOR_R9I_C_ISOTROPIC_TRAINREAL_DATASET_BUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-B status: {b.get('status')}"
        )

    if (
        c.get("status")
        !=
        "R9I_C_V2_MATCHED_ISOTROPIC_TRAINREAL_CANONICAL_DATASETS_BUILT_READY_FOR_R9I_D_MATCHED_NEURAL_RETRAINING_SCRIPT"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-C status: {c.get('status')}"
        )

    source_map = b[
        "original_training_sources"
    ]

    expected_t = source_map[
        "Transformer"
    ][
        "sha256"
    ]

    expected_r = source_map[
        "GRU_LSTM"
    ][
        "sha256"
    ]

    if sha256_file(
        TRANSFORMER_SRC
    ) != expected_t:
        raise RuntimeError(
            "train_transformer.py hash changed after R9-I-B lock."
        )

    if sha256_file(
        RECURRENT_SRC
    ) != expected_r:
        raise RuntimeError(
            "train_lstm_gru.py hash changed after R9-I-B lock."
        )

    return (
        b,
        c,
    )


def parse_source(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(path)
    )

    return (
        text,
        lines,
        tree,
    )


def get_top_level_path_assignments(path):
    text, lines, tree = parse_source(
        path
    )

    rows = []

    for node in tree.body:
        target = None
        value = None

        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
            value = node.value

        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        ):
            target = node.target.id
            value = node.value

        if target is None or value is None:
            continue

        start = int(
            node.lineno
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        )

        source = "\n".join(
            lines[
                start - 1:
                end
            ]
        )

        low = source.lower()

        if not any(
            token in low
            for token in [
                "path(",
                "root",
                "out",
                "data",
                "result",
                "dataset",
                "transformer_results",
                "lstm_gru_results",
            ]
        ):
            continue

        rows.append(
            {
                "file":
                    path.name,

                "name":
                    target,

                "start_line":
                    start,

                "end_line":
                    end,

                "source":
                    "\n".join(
                        f"{i}: {lines[i-1]}"
                        for i in range(
                            start,
                            end + 1
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def get_function_node(tree, name):
    for node in tree.body:
        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                )
            )
            and node.name == name
        ):
            return node

    return None


def signature_text(node):
    args = []

    pos = list(
        node.args.posonlyargs
    ) + list(
        node.args.args
    )

    defaults = [
        None
    ] * (
        len(pos)
        -
        len(node.args.defaults)
    ) + list(
        node.args.defaults
    )

    for arg, default in zip(
        pos,
        defaults
    ):
        if default is None:
            args.append(
                arg.arg
            )
        else:
            try:
                d = ast.unparse(
                    default
                )
            except Exception:
                d = "<default>"

            args.append(
                f"{arg.arg}={d}"
            )

    if node.args.vararg:
        args.append(
            f"*{node.args.vararg.arg}"
        )

    for arg, default in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults
    ):
        if default is None:
            args.append(
                arg.arg
            )
        else:
            try:
                d = ast.unparse(
                    default
                )
            except Exception:
                d = "<default>"

            args.append(
                f"{arg.arg}={d}"
            )

    if node.args.kwarg:
        args.append(
            f"**{node.args.kwarg.arg}"
        )

    return (
        f"{node.name}("
        +
        ", ".join(
            args
        )
        +
        ")"
    )


def extract_function_dependencies(path):
    text, lines, tree = parse_source(
        path
    )

    rows = []
    source_rows = []

    targets = TARGET_FUNCTIONS[
        path.name
    ]

    for name in targets:
        node = get_function_node(
            tree,
            name
        )

        if node is None:
            raise RuntimeError(
                f"{path.name}: missing expected function {name}."
            )

        start = int(
            node.lineno
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        )

        source_rows.append(
            {
                "file":
                    path.name,

                "function":
                    name,

                "signature":
                    signature_text(
                        node
                    ),

                "start_line":
                    start,

                "end_line":
                    end,

                "source":
                    "\n".join(
                        f"{i}: {lines[i-1]}"
                        for i in range(
                            start,
                            end + 1
                        )
                    ),
            }
        )

        for sub in ast.walk(
            node
        ):
            if isinstance(
                sub,
                ast.Constant
            ) and isinstance(
                sub.value,
                str
            ):
                value = sub.value
                low = value.lower()

                if any(
                    token in low
                    for token in DATASET_TOKENS
                ):
                    rows.append(
                        {
                            "file":
                                path.name,

                            "function":
                                name,

                            "kind":
                                "dataset_literal",

                            "value":
                                value,

                            "line_number":
                                getattr(
                                    sub,
                                    "lineno",
                                    None
                                ),
                        }
                    )

                elif any(
                    token in low
                    for token in OUTPUT_TOKENS
                ):
                    rows.append(
                        {
                            "file":
                                path.name,

                            "function":
                                name,

                            "kind":
                                "output_literal",

                            "value":
                                value,

                            "line_number":
                                getattr(
                                    sub,
                                    "lineno",
                                    None
                                ),
                        }
                    )

            if isinstance(
                sub,
                ast.Call
            ):
                call_name = ""

                if isinstance(
                    sub.func,
                    ast.Name
                ):
                    call_name = sub.func.id

                elif isinstance(
                    sub.func,
                    ast.Attribute
                ):
                    try:
                        call_name = ast.unparse(
                            sub.func
                        )
                    except Exception:
                        call_name = sub.func.attr

                if any(
                    token in call_name.lower()
                    for token in [
                        "np.load",
                        "read_csv",
                        "torch.save",
                        "to_csv",
                        "savez",
                        "write_text",
                    ]
                ):
                    try:
                        expr = ast.unparse(
                            sub
                        )
                    except Exception:
                        expr = call_name

                    rows.append(
                        {
                            "file":
                                path.name,

                            "function":
                                name,

                            "kind":
                                "io_call",

                            "value":
                                expr,

                            "line_number":
                                getattr(
                                    sub,
                                    "lineno",
                                    None
                                ),
                        }
                    )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            source_rows
        ),
    )


def get_module_main_calls(path):
    text, lines, tree = parse_source(
        path
    )

    rows = []

    for node in tree.body:
        if not isinstance(
            node,
            ast.If
        ):
            continue

        try:
            test = ast.unparse(
                node.test
            )
        except Exception:
            test = ""

        if "__name__" not in test:
            continue

        start = int(
            node.lineno
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        )

        for sub in ast.walk(
            node
        ):
            if not isinstance(
                sub,
                ast.Call
            ):
                continue

            try:
                expr = ast.unparse(
                    sub
                )
            except Exception:
                continue

            rows.append(
                {
                    "file":
                        path.name,

                    "line_number":
                        getattr(
                            sub,
                            "lineno",
                            None
                        ),

                    "call":
                        expr,
                }
            )

        rows.append(
            {
                "file":
                    path.name,

                "line_number":
                    start,

                "call":
                    "MAIN_BLOCK_SOURCE:\n"
                    +
                    "\n".join(
                        f"{i}: {lines[i-1]}"
                        for i in range(
                            start,
                            end + 1
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def inspect_isotropic_dataset():
    rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        for suffix in [
            "birdsai_trainval_canonical.npz",
            "birdsai_trainval_metadata.csv",
        ]:
            p = (
                ISO_DATASET_DIR
                /
                f"{task}_{suffix}"
            )

            require(
                p
            )

            rows.append(
                {
                    "task":
                        task,

                    "file":
                        p.name,

                    "path":
                        str(
                            p
                        ),

                    "sha256":
                        sha256_file(
                            p
                        ),

                    "size_bytes":
                        int(
                            p.stat().st_size
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-I-D0 — ORIGINAL TRAINING RUNTIME DATA-DEPENDENCY AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. No training and no inference. "
        "The matched isotropic TrainReal dataset exists, but before launching the original "
        "training scripts we must ensure their post-training evaluation does not silently read "
        "old anisotropic BIRDSAI TestReal/WildBox files. This audit extracts path assignments, "
        "train/evaluate function signatures, I/O calls and module main-block calls.\n"
    )

    verify_locks()

    path_frames = []
    dep_frames = []
    func_frames = []
    main_frames = []

    for src in SOURCES:
        path_frames.append(
            get_top_level_path_assignments(
                src
            )
        )

        deps, funcs = extract_function_dependencies(
            src
        )

        dep_frames.append(
            deps
        )

        func_frames.append(
            funcs
        )

        main_frames.append(
            get_module_main_calls(
                src
            )
        )

    path_df = pd.concat(
        path_frames,
        ignore_index=True
    )

    dep_df = pd.concat(
        dep_frames,
        ignore_index=True
    )

    func_df = pd.concat(
        func_frames,
        ignore_index=True
    )

    main_df = pd.concat(
        main_frames,
        ignore_index=True
    )

    iso_inventory = inspect_isotropic_dataset()

    path_csv = (
        OUT
        / "R9I_D0_TOP_LEVEL_PATH_ASSIGNMENTS.csv"
    )

    dep_csv = (
        OUT
        / "R9I_D0_FUNCTION_IO_DEPENDENCIES.csv"
    )

    func_csv = (
        OUT
        / "R9I_D0_TARGET_FUNCTION_SOURCES.csv"
    )

    main_csv = (
        OUT
        / "R9I_D0_MAIN_BLOCK_CALLS.csv"
    )

    iso_csv = (
        OUT
        / "R9I_D0_ISOTROPIC_TRAINVAL_INPUT_INVENTORY.csv"
    )

    path_df.to_csv(
        path_csv,
        index=False
    )

    dep_df.to_csv(
        dep_csv,
        index=False
    )

    func_df.to_csv(
        func_csv,
        index=False
    )

    main_df.to_csv(
        main_csv,
        index=False
    )

    iso_inventory.to_csv(
        iso_csv,
        index=False
    )

    combined_text = "\n".join(
        dep_df[
            "value"
        ].astype(str).tolist()
    ).lower()

    reads_test = (
        "birdsai_test_canonical"
        in
        combined_text
    )

    reads_wild = (
        "wildbox_external_canonical"
        in
        combined_text
    )

    status = (
        "R9I_D0_TRAINING_RUNTIME_DEPENDENCIES_AUDITED_READY_FOR_R9I_D1_SAFE_TRAINING_ONLY_MATCHED_RUNNER"
    )

    manifest = {
        "step":
            "R9I_D0_ORIGINAL_TRAINING_RUNTIME_DATA_DEPENDENCY_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "original_training_sources": {
            "transformer":
                {
                    "path":
                        str(
                            TRANSFORMER_SRC
                        ),

                    "sha256":
                        sha256_file(
                            TRANSFORMER_SRC
                        ),
                },

            "recurrent":
                {
                    "path":
                        str(
                            RECURRENT_SRC
                        ),

                    "sha256":
                        sha256_file(
                            RECURRENT_SRC
                        ),
                },
        },

        "matched_isotropic_trainval_directory":
            str(
                ISO_DATASET_DIR
            ),

        "target_function_signatures":
            func_df[
                [
                    "file",
                    "function",
                    "signature",
                ]
            ].to_dict(
                orient="records"
            ),

        "evaluation_dependency_flags": {
            "birdsai_test_canonical_detected":
                bool(
                    reads_test
                ),

            "wildbox_external_canonical_detected":
                bool(
                    reads_wild
                ),
        },

        "new_training":
            False,

        "model_inference":
            False,

        "frozen_results_modified":
            False,

        "outputs": {
            "path_assignments":
                str(
                    path_csv
                ),

            "function_io_dependencies":
                str(
                    dep_csv
                ),

            "target_function_sources":
                str(
                    func_csv
                ),

            "main_block_calls":
                str(
                    main_csv
                ),

            "isotropic_trainval_inventory":
                str(
                    iso_csv
                ),
        },
    }

    (
        OUT
        / "R9I_D0_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TOP-LEVEL PATH ASSIGNMENTS:"
    )

    print(
        path_df.to_string(
            index=False
        )
    )

    print(
        "\nTRAIN / EVALUATE FUNCTION SIGNATURES:"
    )

    print(
        func_df[
            [
                "file",
                "function",
                "signature",
                "start_line",
                "end_line",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nFUNCTION I/O DEPENDENCIES:"
    )

    print(
        dep_df.to_string(
            index=False
        )
    )

    print(
        "\nMODULE MAIN-BLOCK CALLS:"
    )

    print(
        main_df.to_string(
            index=False
        )
    )

    print(
        "\nISOTROPIC TRAINVAL INPUT INVENTORY:"
    )

    print(
        iso_inventory.to_string(
            index=False
        )
    )

    print(
        "\nEVALUATION DEPENDENCY FLAGS:"
    )

    print(
        json.dumps(
            manifest[
                "evaluation_dependency_flags"
            ],
            indent=2
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-I-D0 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
