"""Offline checks for the v4.5.0 direct-analysis skill."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "drocat-usage"


def test_skill_bundle_has_required_entrypoints_and_references() -> None:
    skill_file = SKILL / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")

    assert text.startswith("---\nname: drocat-usage\n")
    assert "DROCAT v4.5.0" in text
    assert (SKILL / "agents" / "openai.yaml").is_file()
    assert (SKILL / "scripts" / "run_direct.py").is_file()
    for reference in (
        "tool-catalog.md",
        "workflow-recipes.md",
        "datasets-and-auth.md",
        "deepseek-codex.md",
    ):
        assert (SKILL / "references" / reference).is_file()


def test_direct_launcher_dry_run_resolves_script_directory() -> None:
    launcher = SKILL / "scripts" / "run_direct.py"
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--repo",
            str(ROOT),
            "--script",
            "scripts/FindPath.py",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[DROCAT] cwd:" in result.stdout
    assert str(ROOT / "scripts") in result.stdout
    assert str(ROOT / "scripts" / "FindPath.py") in result.stdout


def test_readme_exposes_direct_analysis_and_beginner_setup() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "skills/drocat-usage/SKILL.md" in readme
    assert "docs/INSTALLATION.md" in readme
    assert "deepseek-v4-flash" in readme
    assert "raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-usage/SKILL.md" in readme
    assert "finish the requested analysis end-to-end" in readme


def test_skill_installation_uses_agent_command_not_manual_copy() -> None:
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    setup = (ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")
    raw_url = "raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-usage/SKILL.md"

    assert raw_url in skill
    assert raw_url in setup
    assert "mkdir -p ~/.codex/skills/drocat-usage" not in skill
    assert "mkdir -p ~/.codex/skills/drocat-usage" not in setup


def test_agent_workflows_have_completion_contracts() -> None:
    usage = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    install = (ROOT / "skills" / "drocat-install" / "SKILL.md").read_text(encoding="utf-8")
    setup = (ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")
    deepseek = (SKILL / "references" / "deepseek-codex.md").read_text(encoding="utf-8")

    assert "## Completion contract" in usage
    assert "Do not stop after fetching this skill" in usage
    assert "## Completion contract" in install
    assert "Do not stop after cloning the" in install
    assert "finish the requested analysis end-to-end" in setup
    assert "finish by reporting the validated artifacts" in deepseek
