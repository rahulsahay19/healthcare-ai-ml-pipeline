import json
from pathlib import Path

def test_feature_schema_exists():
    schema_path = Path("outputs/feature_schema.json")
    assert schema_path.exists(), "feature_schema.json doesn't exist"


def test_feature_schema_is_valid_json():
    schema_path = Path("outputs/feature_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "feature_schema.json must contain a JSON object"

def test_feature_schema_has_content():
    schema_path = Path("outputs/feature_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) > 0, "feature_schema.json is empty"
