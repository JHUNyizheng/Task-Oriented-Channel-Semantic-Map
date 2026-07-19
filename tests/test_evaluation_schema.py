from __future__ import annotations

import csv
import json

from tcsm_rt.evaluation import RAW_FIELDS, _archive_incompatible_raw_schema


def test_evaluation_schema_archives_incompatible_raw_file(tmp_path):
    raw = tmp_path / "evaluation_raw.csv"
    raw.write_text("scene_id,model\nscene,ours\n", encoding="utf-8")
    backup = _archive_incompatible_raw_schema(raw)
    assert backup is not None and backup.exists()
    assert not raw.exists()
    manifest = json.loads((tmp_path / "evaluation_schema_migration.json").read_text())
    assert manifest["old_header"] == ["scene_id", "model"]
    assert manifest["new_header"] == list(RAW_FIELDS)


def test_evaluation_schema_preserves_matching_file(tmp_path):
    raw = tmp_path / "evaluation_raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(RAW_FIELDS)
    assert _archive_incompatible_raw_schema(raw) is None
    assert raw.exists()
