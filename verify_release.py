"""Verify the published package without importing analysis code or private data."""

from pathlib import Path
import ast
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "FILE_MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    python_files = json_files = 0
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Missing released file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Changed released file: {name}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=name)
            python_files += 1
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            json_files += 1
    print(json.dumps({
        "status": "passed",
        "released_files_checked": len(manifest["files"]),
        "python_files_parsed": python_files,
        "json_files_parsed": json_files,
        "scope": "File integrity and syntax; no analytical code execution or model fitting.",
    }, indent=2))


if __name__ == "__main__":
    main()
