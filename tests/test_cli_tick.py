from pathlib import Path

from typer.testing import CliRunner

from orgpulse.cli import app


def test_cli_tick_bootstrap_then_remembered(tmp_path: Path):
    mem = tmp_path / "mem.json"
    out = tmp_path / "runs"
    runner = CliRunner()

    r0 = runner.invoke(app, ["tick", "--tick-number", "0", "--memory", str(mem), "--out", str(out), "--offline"])
    assert r0.exit_code == 0, r0.output
    assert mem.exists() and (out / "latest" / "findings.json").exists()
    assert "bootstrapped" in r0.output

    r1 = runner.invoke(app, ["tick", "--tick-number", "1", "--memory", str(mem), "--out", str(out), "--offline"])
    assert r1.exit_code == 0, r1.output
    assert "remembered" in r1.output
