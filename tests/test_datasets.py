import json
from pathlib import Path

from code_explainer.data.datasets import (
    build_dataset_dict,
    DatasetConfig,
    load_from_json,
)


def test_build_dataset_dict_defaults():
    parts = build_dataset_dict(DatasetConfig())
    assert "train" in parts and "eval" in parts
    assert isinstance(parts["train"], list) and len(parts["train"]) >= 1


def test_load_from_json_roundtrip(tmp_path: Path):
    data = [
        {"code": "a=1", "explanation": "sets a to 1"},
        {"code": "b=2", "explanation": "sets b to 2"},
    ]
    p = tmp_path / "d.json"
    p.write_text(json.dumps(data))
    out = load_from_json(str(p))
    assert out == data


def test_build_dataset_with_files(tmp_path: Path):
    train = tmp_path / "train.json"
    evalf = tmp_path / "eval.json"
    train.write_text(json.dumps([{"code": "x=1", "explanation": "sets"}]))
    evalf.write_text(json.dumps([{"code": "y=2", "explanation": "sets"}]))
    parts = build_dataset_dict(
        DatasetConfig(train_file=str(train), eval_file=str(evalf), max_examples=10)
    )
    assert len(parts["train"]) == 1
    assert len(parts["eval"]) == 1
