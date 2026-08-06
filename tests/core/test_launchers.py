"""Structural tests for the launchers and installers: the unified
run_DROCAT.* files, the packed installers, and the token workflow they
implement. These guard the first-run / self-healing / token UX contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestRunDrocatLaunchers:
    def test_macos_launcher_is_self_healing(self):
        text = (ROOT / "run_DROCAT.command").read_text(encoding="utf-8")
        # installs on demand from the packed location
        assert "archive/install/install.sh" in text
        assert "Conda is not installed" in text
        # port-conflict prompt with the DROCAT-ownership guard
        assert "Your choice [1-3]" in text
        assert "Kill the existing DROCAT process" in text
        assert "is not DROCAT" in text
        # token reminder for users who skipped the installer prompt
        assert "token_info_local.txt" in text
        assert "tokens are not configured yet" in text
        # double-click UX: keep the window open on failure
        assert "Press Return to close" in text

    def test_windows_launcher_mirrors(self):
        text = (ROOT / "run_DROCAT.bat").read_text(encoding="utf-8")
        assert "archive\\install\\install.ps1" in text
        assert "Your choice [1-3]" in text
        assert "taskkill /PID" in text
        assert "netstat -ano" in text
        assert "tokens are not configured yet" in text
        assert "pause" in text


class TestInstallers:
    def test_install_sh_resolves_project_root_and_verifies(self):
        text = (ROOT / "archive/install/install.sh").read_text(encoding="utf-8")
        assert 'PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"' in text
        assert 'verify_install.py --project "$PROJECT_ROOT"' in text
        assert 'cd "$PROJECT_ROOT"' in text
        assert "run_DROCAT.command" in text

    def test_install_sh_token_menu_lists_three_ways(self):
        text = (ROOT / "archive/install/install.sh").read_text(encoding="utf-8")
        assert "1. Paste them here in the terminal now" in text
        assert "2. Set them later in the UI Settings tab" in text
        assert "3. Edit token_info_local.txt manually" in text
        assert "Non-interactive: skipping the token prompt" in text
        assert "Skipped: no tokens written" in text
        assert "Nothing to write: tokens are already configured" in text
        # a full skip must not create a half-configured file
        assert "Only write when something was entered" in text

    def test_install_ps1_mirrors_token_menu(self):
        text = (ROOT / "archive/install/install.ps1").read_text(encoding="utf-8")
        assert "[Console]::IsInputRedirected" in text
        assert "Paste them here in the terminal now" in text
        assert "Skipped: no tokens written" in text
        assert "Nothing to write: tokens are already configured" in text
        assert "run_DROCAT.bat" in text

    def test_install_bat_wraps_ps1(self):
        text = (ROOT / "archive/install/install.bat").read_text(encoding="utf-8")
        assert "install.ps1" in text
        assert "-ExecutionPolicy Bypass" in text
