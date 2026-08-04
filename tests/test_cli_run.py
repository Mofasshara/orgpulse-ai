import json
from pathlib import Path

from typer.testing import CliRunner

from orgpulse.cli import app, write_run


def test_write_run_offline_writes_artifacts(tmp_path: Path):
    f = write_run(out=tmp_path, days=200, seed=42, offline=True)
    findings = json.loads((tmp_path / "findings.json").read_text())
    assert findings["run_id"] and "recommendations" in findings
    assert (tmp_path / "report.md").exists()
    assert f.overall_severity in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}


def test_cli_run_offline(tmp_path: Path):
    result = CliRunner().invoke(app, ["run", "--days", "200", "--out", str(tmp_path), "--offline"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "findings.json").exists() and (tmp_path / "report.md").exists()
