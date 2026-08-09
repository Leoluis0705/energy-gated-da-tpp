from __future__ import annotations

from pathlib import Path

from analysis.build_audit_output_manifest import collect_audit_files


def test_manifest_collection_is_sorted_and_excludes_runtime_products(tmp_path: Path) -> None:
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis/a.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "analysis/__pycache__").mkdir()
    (tmp_path / "analysis/__pycache__/a.pyc").write_bytes(b"ignored")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/z.md").write_text("z\n", encoding="utf-8")
    (tmp_path / "docs/audit_output_sha256.csv").write_text("self\n", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results/a.csv").write_text("a\n", encoding="utf-8")
    frame = collect_audit_files(tmp_path)
    assert frame["path"].tolist() == ["analysis/a.py", "docs/z.md", "results/a.csv"]
    assert frame["size_bytes"].gt(0).all()
    assert frame["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
