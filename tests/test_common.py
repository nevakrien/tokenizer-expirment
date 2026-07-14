import json

import pytest

from experiment.common import load_document_records, load_documents


def test_load_jsonl_documents(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps({"text": "first"}) + "\n" + json.dumps("second") + "\n",
        encoding="utf-8",
    )

    assert load_documents([str(path)]) == [b"first", b"second"]
    records = load_document_records([str(path)])
    assert [record.path for record in records] == [f"{path}:1", f"{path}:2"]


def test_load_jsonl_rejects_missing_text(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps({"title": "missing"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="string 'text' field"):
        load_documents([str(path)])
