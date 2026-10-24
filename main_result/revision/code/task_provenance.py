from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


class CacheProvenanceError(RuntimeError):
    pass


def file_record(path):
    path = Path(path)
    if not path.is_file():
        raise CacheProvenanceError(f"Required provenance file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def canonical(value):

    def normalize(item):
        if isinstance(item, float) and (not math.isfinite(item)):
            return {
                "__nonfinite_float__": (
                    "nan" if math.isnan(item) else "+inf" if item > 0 else "-inf"
                )
            }
        if isinstance(item, dict):
            return {key: normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        return item

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_spec(task, inputs, code, settings, libraries):
    code = dict(code, task_provenance=Path(__file__))
    spec = {
        "schema_version": 1,
        "task": task,
        "inputs": {name: file_record(path) for name, path in sorted(inputs.items())},
        "code": {name: file_record(path) for name, path in sorted(code.items())},
        "settings": settings,
        "libraries": libraries,
    }
    return json.loads(canonical(spec))


def _reject(receipt, reason):
    raise CacheProvenanceError(
        f"Cache reuse refused for {receipt}: {reason}. Preserve the existing run and rerun fitting in a fresh output directory. Legacy or mismatched outputs are never adopted automatically."
    )


def reusable(receipt, spec, outputs):
    receipt = Path(receipt)
    present = {name: Path(path).exists() for name, path in outputs.items()}
    if not receipt.exists():
        if any(present.values()):
            _reject(receipt, "outputs exist but their provenance receipt is missing")
        return False
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _reject(receipt, f"unreadable provenance receipt ({exc})")
    if (
        not isinstance(record, dict)
        or record.get("status") != "complete"
        or record.get("spec") != spec
    ):
        _reject(receipt, "input, code, grid, parameters, seed, or library fingerprint differs")
    fingerprint = hashlib.sha256(canonical(spec).encode("utf-8")).hexdigest()
    if record.get("fingerprint_sha256") != fingerprint:
        _reject(receipt, "receipt fingerprint is inconsistent")
    recorded_outputs = record.get("outputs")
    if not isinstance(recorded_outputs, dict) or set(recorded_outputs) != set(outputs):
        _reject(receipt, "output list differs from the required task outputs")
    for name, path in outputs.items():
        if not Path(path).is_file() or file_record(path) != recorded_outputs[name]:
            _reject(receipt, f"output is missing or its hash changed ({name})")
    return True


def write_receipt(receipt, spec, outputs):
    receipt = Path(receipt)
    if receipt.exists():
        _reject(receipt, "a receipt already exists; refusing to replace run provenance")
    records = {name: file_record(path) for name, path in sorted(outputs.items())}
    payload = {
        "status": "complete",
        "spec": spec,
        "fingerprint_sha256": hashlib.sha256(canonical(spec).encode("utf-8")).hexdigest(),
        "outputs": records,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=receipt.parent,
            prefix=receipt.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, receipt)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
