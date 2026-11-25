import re
from pathlib import Path

import pytest

from sitefacts.config import repo_root
from sitefacts.messages import build_sft_messages, visible_prompt_text
from sitefacts.pipeline import load_trainable, sft_conversations
from sitefacts.pngutil import read_png_size_and_first_pixel
from sitefacts.records import SiteRecord, load_jsonl, write_jsonl
from sitefacts.sample_catalog import SWATCH_RGB, sample_rows
from sitefacts.sample_data import write_sample_dataset


def test_committed_sample_matches_the_generator(tmp_path: Path):
    write_sample_dataset(tmp_path)
    generated = tmp_path / "data" / "sample"
    committed = repo_root() / "data" / "sample"
    assert (generated / "sites.jsonl").read_text(encoding="utf-8") == (
        committed / "sites.jsonl"
    ).read_text(encoding="utf-8")
    for image in sorted((generated / "images").glob("*.png")):
        assert image.read_bytes() == (committed / "images" / image.name).read_bytes()


def test_splits_and_swatches_are_what_the_labels_say():
    path = repo_root() / "data" / "sample" / "sites.jsonl"
    records = load_jsonl(path)
    assert {record.label_status for record in records} == {"synthetic_sample"}
    assert sum(record.split == "train" for record in records) == 4
    assert sum(record.split == "test" for record in records) == 2
    for record in records:
        assert record.image is not None
        assert record.swatch_color is not None
        width, height, pixel = read_png_size_and_first_pixel(path.parent / record.image)
        assert (width, height) == (32, 32)
        assert pixel == SWATCH_RGB[record.swatch_color]
        assert "SYNTHETIC SAMPLE" in record.notes


def test_prompts_do_not_leak_hidden_answers_or_effort_control():
    root = repo_root() / "data" / "sample"
    for row in sample_rows():
        record = SiteRecord.from_dict(row)
        messages = build_sft_messages(record, root, embed_images=False)
        assert [message["role"] for message in messages][:3] == ["system", "tool_declare", "user"]
        prompt = visible_prompt_text(messages)
        assert "Thinking effort level" not in prompt
        assert record.swatch_color is not None
        assert re.search(rf"\b{re.escape(record.swatch_color)}\b", prompt) is None
        image = messages[2]["content"][0]
        assert image["type"] == "image"
        assert str(image["image"]).endswith(record.image)
    port = next(SiteRecord.from_dict(row) for row in sample_rows() if row["example_id"] == "sample-port-004")
    messages = build_sft_messages(port, root)
    user = messages[2]["content"][1]["text"]
    assert "12.4" not in user
    tool_message = next(message for message in messages if message["role"] == "tool")
    assert "12.4" in tool_message["content"]
    assert messages[-1]["role"] == "assistant"
    assert "12.4" in messages[-1]["content"]


def test_embedded_image_is_a_png_data_uri():
    root = repo_root() / "data" / "sample"
    record = SiteRecord.from_dict(sample_rows()[0])
    messages = build_sft_messages(record, root, embed_images=True)
    image = messages[2]["content"][0]["image"]
    assert image.startswith("data:image/png;base64,")


def test_unlabeled_rows_are_skipped(tmp_path: Path):
    rows = sample_rows()
    rows.append(
        {
            **sample_rows()[0],
            "example_id": "public-tile",
            "split": "unlabeled",
            "label_status": "unlabeled_public",
            "facts": [],
            "latitude": None,
            "longitude": None,
        }
    )
    path = tmp_path / "sites.jsonl"
    write_jsonl(path, [SiteRecord.from_dict(row) for row in rows])
    train = load_trainable(path, "train")
    assert all(record.label_status == "synthetic_sample" for record in train)
    assert all(record.example_id != "public-tile" for record in train)
    empty = tmp_path / "empty.jsonl"
    write_jsonl(
        empty,
        [
            SiteRecord.from_dict(
                {**sample_rows()[0], "split": "unlabeled", "label_status": "unlabeled_public"}
            )
        ],
    )
    with pytest.raises(ValueError, match="unlabeled"):
        load_trainable(empty, "train")


def test_sft_conversations_cover_train():
    path = repo_root() / "data" / "sample" / "sites.jsonl"
    rows = sft_conversations(path, "train")
    assert len(rows) == 4
    assert any(message["role"] == "tool" for _record, messages in rows for message in messages)
