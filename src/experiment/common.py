from __future__ import annotations

import json
from pathlib import Path


def load_documents(input_files: list[str] | None = None, dataset_config: str | None = None) -> list[bytes]:
    paths: list[Path] = []
    if dataset_config:
        config = json.loads(Path(dataset_config).read_text(encoding="utf-8"))
        paths.extend(Path(item) for item in config.get("files", []))
        if "path" in config:
            paths.append(Path(config["path"]))
    if input_files:
        paths.extend(Path(item) for item in input_files)
    if not paths:
        raise ValueError("provide --input-files or --dataset-config")
    documents: list[bytes] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    documents.append(child.read_bytes())
        else:
            documents.append(path.read_bytes())
    return documents
