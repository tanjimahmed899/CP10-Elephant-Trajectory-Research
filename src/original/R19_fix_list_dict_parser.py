from pathlib import Path
import re
import shutil
import sys

TARGET_NAME = "R19_final_claim_matrix_and_paper_rewrite_lock.py"

def main():
    here = Path(__file__).resolve().parent
    target = here / TARGET_NAME

    if not target.exists():
        print(f"[ERROR] Could not find: {target}")
        print(f"Put this fixer in the same folder as {TARGET_NAME} and run it again.")
        sys.exit(1)

    text = target.read_text(encoding="utf-8")

    # If already patched, exit safely.
    if "# R19 SAFE STATUS PARSER — dict/list compatible" in text:
        print("[OK] R19 parser is already patched. No changes made.")
        return

    pattern = re.compile(
        r'^(?P<indent>[ \t]*)status\s*=\s*data\.get\(\s*["\']status["\']\s*,\s*["\']["\']\s*\)\s*$',
        re.MULTILINE,
    )

    match = pattern.search(text)
    if not match:
        print("[ERROR] Exact target line was not found:")
        print('        status = data.get("status", "")')
        print("No file was changed.")
        sys.exit(2)

    indent = match.group("indent")
    i1 = indent + "    "
    i2 = indent + "        "
    i3 = indent + "            "

    replacement = (
        indent + "# R19 SAFE STATUS PARSER — dict/list compatible\n"
        + indent + "if isinstance(data, dict):\n"
        + i1 + 'status = str(data.get("status", "") or data.get("final_status", "") or "")\n'
        + indent + "elif isinstance(data, list):\n"
        + i1 + "status_values = []\n"
        + i1 + "for item in data:\n"
        + i2 + "if isinstance(item, dict):\n"
        + i3 + 'value = item.get("status", "") or item.get("final_status", "")\n'
        + i3 + "if value:\n"
        + i3 + "    status_values.append(str(value))\n"
        + i1 + 'status = " | ".join(status_values)\n'
        + indent + "else:\n"
        + i1 + 'status = ""'
    )

    backup = target.with_suffix(target.suffix + ".bak_before_parser_fix")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[BACKUP] {backup.name}")
    else:
        print(f"[BACKUP] Existing backup kept: {backup.name}")

    patched = text[:match.start()] + replacement + text[match.end():]

    # Compile-check before writing.
    try:
        compile(patched, str(target), "exec")
    except SyntaxError as e:
        print("[ERROR] Patch would create a syntax error. Original file was NOT changed.")
        print(f"{e.__class__.__name__}: {e}")
        sys.exit(3)

    target.write_text(patched, encoding="utf-8")

    print("[SUCCESS] R19 parser patch applied.")
    print(f"[TARGET]  {target}")
    print("No experiment, model, result, or JSON data was rerun or modified.")
    print()
    print("Now run:")
    print(f'python .\\{TARGET_NAME}')

if __name__ == "__main__":
    main()
