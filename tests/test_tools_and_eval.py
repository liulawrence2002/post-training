import ast
import re

from sitefacts.config import repo_root
from sitefacts.fetch_public_samples import unlabeled_record
from sitefacts.offline_eval import score_records
from sitefacts.records import SiteRecord, load_jsonl
from sitefacts.sample_catalog import sample_rows
from sitefacts.tools import SiteContextTool, openai_tool_declarations


def test_tool_stub_returns_only_the_episode_context():
    record = SiteRecord.from_dict(next(row for row in sample_rows() if row["example_id"] == "sample-port-004"))
    tool = SiteContextTool(
        record.site_id,
        record.context,
        source_uri="sample://context/sample-site-port-004",
        observed_at=record.observed_at,
    )
    found = tool.run({"site_id": record.site_id, "fields": ["elevation_m"]})
    assert found["ok"] is True
    assert found["fields"] == {"elevation_m": 12.4}
    missing = tool.run({"site_id": "other-site", "fields": ["elevation_m"]})
    assert missing["ok"] is False
    assert missing["error"] == "unknown_site"
    declared = openai_tool_declarations()
    assert "lookup_site_context" in declared
    assert "12.4" not in declared


def test_offline_canonical_eval_does_not_need_a_key():
    records = [SiteRecord.from_dict(row) for row in sample_rows()]
    report = score_records(records)
    assert report["n"] == 6
    assert report["mean_reward"] == 1.0
    assert report["schema_valid_rate"] == 1.0
    saved = score_records(records, {record.example_id: "not json" for record in records})
    assert saved["mean_reward"] == 0.0


def test_public_sample_record_is_unlabeled():
    row = unlabeled_record("public-modis-2014-04-09", "2014-04-09", "https://example.test/tile.jpg", "images/x.jpg")
    assert row["label_status"] == "unlabeled_public"
    assert row["facts"] == []
    assert "TODO(labels)" in row["notes"]


def _calls_tokenizer_encode(source: str) -> bool:
    """True when the module calls ``tokenizer.encode``, ignoring prose that forbids it."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "encode":
            continue
        value = node.func.value
        if isinstance(value, ast.Name) and value.id == "tokenizer":
            return True
    return False


def test_training_sources_keep_inkling_rules():
    root = repo_root() / "src" / "sitefacts"
    rendering = (root / "rendering.py").read_text(encoding="utf-8")
    assert "get_recommended_renderer_name" in rendering
    assert _calls_tokenizer_encode(rendering) is False
    assert "get_lr" in rendering
    for name in ("train_sft.py", "train_rl.py", "evaluate.py", "rl_env.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert _calls_tokenizer_encode(text) is False
        assert "effort=" in text or "effort:" in text
        ast.parse(text)
    rl = (root / "train_rl.py").read_text(encoding="utf-8")
    assert "remove_constant_reward_groups" in rl
    assert "0.1" in (root / "rendering.py").read_text(encoding="utf-8") or "scale" in rendering
    committed = load_jsonl(repo_root() / "data" / "sample" / "sites.jsonl")
    assert committed


def test_repo_has_no_api_key():
    banned_suffixes = {".png", ".jpg", ".jpeg"}
    for path in repo_root().rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix in banned_suffixes:
            continue
        if path.stat().st_size > 1_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = re.match(r"(?:export\s+)?TINKER_API_KEY\s*=\s*(.*)$", stripped)
            if match is None:
                continue
            value = match.group(1).strip().strip("'\"")
            assert value == "", f"{path} contains a TINKER_API_KEY value"
