"""Copy MELP's CKEPE prompts into a versioned local zero-shot asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from zeroshot.data.tasks import TASKS

DEFAULT_SOURCE = Path(
    "../MELP-main/src/melp/prompt/CKEPE_prompt.json"
)
DEFAULT_OUTPUT = Path(__file__).with_name("melp_ckepe.json")
DEFAULT_AUDIT = Path(__file__).with_name("melp_ckepe.audit.json")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def prepare(source: Path, output: Path, audit_path: Path) -> None:
    source = source.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = sorted(
        {label for task in TASKS.values() for label in task.labels}
    )
    missing = [label for label in required if label not in payload]
    if missing:
        raise KeyError(f"Source CKEPE prompts are missing {missing}")
    invalid = [
        label
        for label in required
        if not isinstance(payload[label], str) or not payload[label].strip()
    ]
    if invalid:
        raise ValueError(f"Empty/non-string prompts: {invalid}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    audit = {
        "source": str(source),
        "source_sha256": sha256(source),
        "snapshot": str(output.resolve()),
        "snapshot_sha256": sha256(output),
        "entry_count": len(payload),
        "required_label_count": len(required),
        "missing_required_labels": [],
    }
    temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_audit.replace(audit_path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    prepare(args.source, args.output, args.audit)
    print(f"Wrote {args.output} and {args.audit}")

if __name__ == "__main__":
    main()
