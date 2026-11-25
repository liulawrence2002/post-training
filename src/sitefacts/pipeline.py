"""Load geospatial episodes and turn them into multimodal SFT conversations.

The committed dataset is synthetic. Public downloads are unlabeled and are
skipped here. Rendering to tokens happens in the training script, with an
explicit thinking effort, and is not cached to disk: tokenizers differ by
Inkling model, so a rendered dataset must not be reused across model names.
"""

from __future__ import annotations

from pathlib import Path

from sitefacts.messages import build_sft_messages
from sitefacts.records import SiteRecord, load_jsonl, split_records


def load_trainable(path: Path, split: str) -> list[SiteRecord]:
    records = load_jsonl(path)
    kept, _skipped = split_records(records, split)
    if not kept:
        raise ValueError(
            f"No trainable {split} rows in {path}. "
            "Rows need label_status 'synthetic_sample' and a train or test split. "
            "Unlabeled public samples are skipped on purpose."
        )
    return kept


def data_root_for(path: Path) -> Path:
    """Directory that relative image paths are resolved against."""
    return path.parent


def sft_conversations(
    path: Path,
    split: str,
    *,
    embed_images: bool = False,
) -> list[tuple[SiteRecord, list[dict]]]:
    root = data_root_for(path)
    rows = []
    for record in load_trainable(path, split):
        rows.append((record, build_sft_messages(record, root, embed_images=embed_images)))
    return rows


def main() -> None:
    """Print the first training conversation as text. Does not call Tinker."""
    from sitefacts.config import repo_root

    path = repo_root() / "data" / "sample" / "sites.jsonl"
    record, messages = sft_conversations(path, "train")[0]
    print(f"example_id={record.example_id} label_status={record.label_status}")
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            preview = " ".join(
                part["text"] if part.get("type") == "text" else f"[{part.get('type')}]"
                for part in content
            )
        else:
            preview = str(content)
        print(f"\n--- {message['role']} ---")
        print(preview[:1200])


if __name__ == "__main__":
    main()
