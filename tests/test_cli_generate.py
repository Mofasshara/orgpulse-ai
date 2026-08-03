from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from orgpulse.cli import app, write_streams


def test_write_streams_creates_three_csvs(tmp_path: Path):
    paths = write_streams(out=tmp_path, days=60, seed=42)
    assert set(paths) == {"clinical", "qc", "supply"}
    for name, p in paths.items():
        assert p.exists(), f"{name} not written"
        df = pd.read_csv(p)
        assert "date" in df.columns and len(df) > 0


def test_cli_generate_runs(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(app, ["generate", "--days", "30", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "clinical.csv").exists()
    assert (tmp_path / "qc.csv").exists()
    assert (tmp_path / "supply.csv").exists()
