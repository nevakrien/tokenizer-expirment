from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    data: bytes
    path: str
    group: str | None = None


def load_documents(input_files: list[str] | None = None, dataset_config: str | None = None) -> list[bytes]:
    return [record.data for record in load_document_records(input_files, dataset_config)]


def load_document_records(input_files: list[str] | None = None, dataset_config: str | None = None) -> list[DocumentRecord]:
    paths: list[Path] = []
    grouped_paths: list[tuple[Path, str | None]] = []
    if dataset_config:
        config = json.loads(Path(dataset_config).read_text(encoding="utf-8"))
        for item in config.get("files", []):
            path, group = _parse_config_path(item)
            grouped_paths.append((path, group))
        if "path" in config:
            grouped_paths.append((Path(config["path"]), config.get("group") or config.get("language")))
        for group, items in config.get("groups", {}).items():
            for item in items:
                path, item_group = _parse_config_path(item)
                grouped_paths.append((path, item_group or group))
    if input_files:
        paths.extend(Path(item) for item in input_files)
    grouped_paths.extend((path, None) for path in paths)
    if not grouped_paths:
        raise ValueError("provide --input-files or --dataset-config")
    records: list[DocumentRecord] = []
    for path, group in grouped_paths:
        records.extend(_read_records(path, group))
    return records


def _parse_config_path(item: object) -> tuple[Path, str | None]:
    if isinstance(item, str):
        return Path(item), None
    if isinstance(item, dict):
        path = item.get("path") or item.get("file")
        if not path:
            raise ValueError("dataset file entries must include 'path' or 'file'")
        group = item.get("group") or item.get("language") or item.get("lang")
        return Path(path), group
    raise TypeError("dataset file entries must be strings or objects")


def _read_records(path: Path, group: str | None) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                records.extend(_read_records(child, group))
    elif path.suffix == ".jsonl":
        records.extend(_read_jsonl_records(path, group))
    else:
        records.append(DocumentRecord(data=path.read_bytes(), path=str(path), group=group))
    return records


def _read_jsonl_records(path: Path, group: str | None) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = _jsonl_text(item, path, line_number)
            records.append(DocumentRecord(data=text.encode("utf-8"), path=f"{path}:{line_number}", group=group))
    return records


def _jsonl_text(item: Any, path: Path, line_number: int) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        text = item.get("text")
        if isinstance(text, str):
            return text
    raise ValueError(f"{path}:{line_number}: JSONL records must be strings or objects with a string 'text' field")
