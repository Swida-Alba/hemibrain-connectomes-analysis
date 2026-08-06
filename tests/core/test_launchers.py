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
        assert "NeuPrint token is not configured yet" in text
        assert "CAVE token is optional" in text
        # double-click UX: keep the window open on failure
        assert "Press Return to close" in text

    def test_windows_launcher_mirrors(self):
        text = (ROOT / "run_DROCAT.bat").read_text(encoding="utf-8")
        assert "archive\\install\\install.ps1" in text
        assert "Your choice [1-3]" in text
        assert "taskkill /PID" in text
        assert "netstat -ano" in text
        assert "NeuPrint token is not configured yet" in text
        assert "CAVE token is optional" in text
        assert "pause" in text


class TestInstallers:
    def test_install_sh_resolves_project_root_and_verifies(self):
        text = (ROOT / "archive/install/install.sh").read_text(encoding="utf-8")
        assert 'PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"' in text
        assert 'verify_install.py --project "$PROJECT_ROOT"' in text
        assert 'cd "$PROJECT_ROOT"' in text
        assert "run_DROCAT.command" in text

    def test_install_sh_token_notice_has_no_terminal_prompt(self):
        """Tokens are NOT collected in the terminal: the installer only prints
        a status notice pointing to the UI Settings tab / token_info_local.txt,
        with the CAVE token marked optional."""
        text = (ROOT / "archive/install/install.sh").read_text(encoding="utf-8")
        assert "UI Settings tab" in text
        assert "token_info_local.txt" in text
        assert "required for NeuPrint datasets" in text
        assert "only needed for FlyWire FAFB online fetching" in text
        # no interactive paste prompt remains
        assert "read -r -p" not in text
        assert "Paste them here in the terminal" not in text
        assert "Non-interactive: skipping" not in text

    def test_install_ps1_token_notice_has_no_terminal_prompt(self):
        text = (ROOT / "archive/install/install.ps1").read_text(encoding="utf-8")
        assert "UI Settings tab" in text
        assert "required for NeuPrint datasets" in text
        assert "only needed for FlyWire FAFB online fetching" in text
        assert "Read-Host" not in text
        assert "IsInputRedirected" not in text
        assert "Paste them here in the terminal" not in text

    def test_install_bat_wraps_ps1(self):
        text = (ROOT / "archive/install/install.bat").read_text(encoding="utf-8")
        assert "install.ps1" in text
        assert "-ExecutionPolicy Bypass" in text
