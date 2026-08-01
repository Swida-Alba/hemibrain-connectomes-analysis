"""
DROCAT UI Runner Engine
Executes tools by generating proper Python scripts with user parameters.
"""

import asyncio
import os
import sys
import tempfile
import platform
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any

from .config import PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR

# The PlotPath tool lives in the vispath subproject, imported as `vispath_pkg`.
# It is not installed into site-packages, so generated scripts must add
# vispath-subproject/src to sys.path themselves.
VISPATH_DIR = PROJECT_ROOT / "vispath-subproject" / "src"


# =============================================================================
# Tool Registry: Maps tool names to their import paths and method calls
# =============================================================================
TOOL_REGISTRY: Dict[str, dict] = {
    "find_path": {
        "label": "Find All Paths",
        "import": "from coana import FindNeuronConnection",
        "class": "FindNeuronConnection",
        "var": "fc",
        "init_method": "InitializeNeuronInfo",
        "methods": {
            "find_path": "fc.FindPath()",
            "find_all_path": "fc.FindAllPath(forward_only=True)",
        },
    },
    "find_direct": {
        "label": "Find Direct Connections",
        "import": "from coana import FindNeuronConnection",
        "class": "FindNeuronConnection",
        "var": "fc",
        "init_method": "InitializeNeuronInfo",
        "methods": {
            "find_direct": "fc.FindDirectConnections()",
        },
    },
    "connectivity_profiling": {
        "label": "Connectivity Profiling",
        "import": "from comparison.profile_comparator import ConnectivityProfileComparer",
        "class": "ConnectivityProfileComparer",
        "var": "comparer",
        "init_method": None,
        "methods": {
            "run": "comparer.run()",
        },
    },
    "find_homologs": {
        "label": "Homolog Finding",
        "import": "from comparison.profile_comparator import HomologFinder",
        "class": "HomologFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "find_homologs_fast": "finder.find_homologs_fast()",
            "find_homologs": "finder.find_homologs()",
        },
    },
    "inter_dataset": {
        "label": "Cross-Dataset Comparison",
        "import": "from comparison import ComparisonParameters, ComparisonAnalyzer",
        "class": "ComparisonAnalyzer",
        "var": "analyzer",
        "init_method": None,
        "methods": {
            "run": "analyzer.run()",
        },
        "wrapper": True,  # Needs special wrapper for ComparisonParameters
    },
    "nb_find_lines": {
        "label": "Find Driver Lines",
        "import": "from neuronbridge_finder import NeuronBridgeFinder",
        "class": "NeuronBridgeFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "find_lines": "finder.find_lines_batch(**method_params)",
        },
    },
    "nb_find_neuron": {
        "label": "Find EM Neurons",
        "import": "from neuronbridge_finder import NeuronBridgeFinder",
        "class": "NeuronBridgeFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "find_neurons": "finder.find_neurons_batch(**method_params)",
        },
    },
    "nb_colabel": {
        "label": "Co-Labeling Analysis",
        "import": "from neuronbridge_finder import NeuronBridgeFinder",
        "class": "NeuronBridgeFinder",
        "var": "finder",
        "init_method": None,
        "methods": {
            "colabel": "finder.analyze_colabeling(**method_params)",
        },
    },
    "plot3d_skeleton": {
        "label": "3D Skeleton Visualization",
        "import": "from visualize_skeleton import VisualizeSkeleton",
        "class": "VisualizeSkeleton",
        "var": "vs",
        "init_method": None,
        "methods": {
            "plot": "vs.plot_neurons()",
        },
    },
    "plot_path": {
        "label": "Path Network Visualization",
        "import": "from vispath_pkg import VisualizePath",
        "class": "VisualizePath",
        "var": "vp",
        "init_method": None,
        "methods": {
            "plot": "vp.visualize()",
        },
    },
}


def _format_value(value: Any) -> str:
    """Format a Python value for code generation."""
    if isinstance(value, str):
        return repr(value)
    elif isinstance(value, bool):
        return repr(value)
    elif value is None:
        return "None"
    elif isinstance(value, list):
        return repr(value)
    elif isinstance(value, dict):
        return repr(value)
    else:
        return str(value)


def _format_params(params: dict) -> str:
    """Format a params dict into constructor argument string."""
    lines = []
    for key, value in params.items():
        lines.append(f"    {key}={_format_value(value)},")
    return "\n".join(lines)


class ScriptRunner:
    """Runs DROCAT tools as subprocesses with real-time output streaming."""

    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.is_running = False
        self._cancelled = False

    async def run(
        self,
        tool_name: str,
        constructor_params: dict,
        method_name: str = "run",
        method_params: Optional[dict] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run a tool with given parameters.

        Args:
            tool_name: Key in TOOL_REGISTRY
            constructor_params: Parameters for the class constructor
            method_name: Key in tool's 'methods' dict
            method_params: Parameters for the method call (if any)
            log_callback: Callback(line, stream) for log output
            output_dir: Expected output directory to scan for results

        Returns:
            dict with 'returncode', 'files', 'duration'
        """
        if tool_name not in TOOL_REGISTRY:
            msg = f"Unknown tool: {tool_name}"
            if log_callback:
                log_callback(msg, "error")
            return {"returncode": -1, "files": [], "duration": 0, "cancelled": False}

        tool = TOOL_REGISTRY[tool_name]
        self.is_running = True
        self._cancelled = False
        start_time = datetime.now()

        # Generate the script
        script_content = self._generate_script(
            tool_name, constructor_params, method_name, method_params
        )

        # Write to temp file
        fd, temp_script = tempfile.mkstemp(suffix=".py", prefix="drocat_run_")
        with os.fdopen(fd, "w") as f:
            f.write(script_content)

        try:
            if log_callback:
                label = tool.get("label", tool_name)
                log_callback("", "system")
                log_callback(
                    f"━━━ ▶ UI FUNCTION: {label}  ({tool_name}) ━━━", "system"
                )
                log_callback(
                    "Output streams live below while the function runs; "
                    "generated files appear in Output Files when it finishes.",
                    "system",
                )
                log_callback("", "system")
                # Log params as multi-line for readability
                log_callback("[DROCAT] Parameters:", "system")
                for k, v in constructor_params.items():
                    log_callback(f"  {k}: {v}", "system")
                if method_params:
                    for k, v in method_params.items():
                        log_callback(f"  {k}: {v}", "system")

            # Get Python executable
            python_exe = sys.executable or "python"

            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + str(PROJECT_ROOT)
            # Run unbuffered so log lines stream to the UI in real time
            env["PYTHONUNBUFFERED"] = "1"

            # Create subprocess
            self.process = await asyncio.create_subprocess_exec(
                python_exe,
                "-u",
                temp_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                env=env,
            )

            # Stream output
            if log_callback:
                await self._stream_output(log_callback)

            # Wait for completion
            returncode = await self.process.wait()

            duration = (datetime.now() - start_time).total_seconds()

            if self._cancelled:
                if log_callback:
                    log_callback("[DROCAT] Execution cancelled by user.", "error")
            elif returncode == 0:
                if log_callback:
                    log_callback(
                        f"━━━ ■ FINISHED: {tool.get('label', tool_name)} — "
                        f"completed in {duration:.1f}s ━━━",
                        "success",
                    )
            else:
                if log_callback:
                    log_callback(
                        f"━━━ ■ FINISHED: {tool.get('label', tool_name)} — "
                        f"failed with return code {returncode} ━━━",
                        "error",
                    )

            # Scan output directory for files
            files = self._scan_output_files(output_dir) if output_dir else []

            return {
                "returncode": returncode,
                "files": files,
                "duration": duration,
                "cancelled": self._cancelled,
            }

        except Exception as e:
            if log_callback:
                log_callback(f"[DROCAT] Error: {str(e)}", "error")
            return {"returncode": -1, "files": [], "duration": 0, "cancelled": False}
        finally:
            self.is_running = False
            self.process = None
            # Clean up temp file
            try:
                os.unlink(temp_script)
            except OSError:
                pass

    def _generate_script(
        self,
        tool_name: str,
        constructor_params: dict,
        method_name: str,
        method_params: Optional[dict],
    ) -> str:
        """Generate a Python script that runs the tool with given parameters."""
        tool = TOOL_REGISTRY[tool_name]
        import_stmt = tool["import"]
        class_name = tool["class"]
        var = tool["var"]
        init_method = tool.get("init_method")
        method_call = tool["methods"].get(method_name, "")

        # Handle special wrapper cases
        if tool_name == "inter_dataset":
            return self._generate_inter_dataset_script(constructor_params, method_params)
        elif tool_name in ("nb_find_lines", "nb_find_neuron", "nb_colabel"):
            return self._generate_neuronbridge_script(tool_name, constructor_params, method_params)

        # Standard script generation
        params_str = _format_params(constructor_params)

        init_call = ""
        if init_method:
            init_call = f"{var}.{init_method}()\n"

        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for {tool_name}."""
import sys
import warnings
from pathlib import Path

# Add project paths
sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

# Import the tool class
{import_stmt}

# Create instance with user parameters
{var} = {class_name}(
{params_str}
)

# Initialize (if needed)
{init_call}# Run the method
{method_call}

print("[DROCAT] Done.")
'''
        return script

    def _generate_inter_dataset_script(
        self, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate script for InterDatasetComparator."""
        params_str = _format_params(constructor_params)
        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for inter-dataset comparison."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
{params_str}
)

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.generate_report()

print("[DROCAT] Done.")
'''
        return script

    def _generate_neuronbridge_script(
        self, tool_name: str, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate script for NeuronBridge tools."""
        init_params_str = _format_params(constructor_params)
        method_params_str = _format_params(method_params or {})

        tool = TOOL_REGISTRY[tool_name]
        method_name = "find_lines_batch" if tool_name == "nb_find_lines" else \
                       "find_neurons_batch" if tool_name == "nb_find_neuron" else "analyze_colabeling"

        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for {tool_name}."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

from neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(
{init_params_str}
)

finder.{method_name}(
{method_params_str}
)

print("[DROCAT] Done.")
'''
        return script

    async def _stream_output(self, log_callback: Callable[[str, str], None]):
        """
        Stream stdout/stderr to the UI in real time.

        Reads chunks as soon as they arrive (unbuffered subprocess), emits
        complete lines immediately, and forwards carriage-return progress
        (e.g. tqdm bars) as a live-updating 'progress' line so the log never
        appears frozen while a long-running function works.
        """
        if not self.process:
            return

        async def read_stream(stream, stream_name):
            buffer = ""
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                # Emit every complete line immediately
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.rstrip("\r")
                    if line:
                        log_callback(line, stream_name)

                # A trailing carriage return means an in-place progress update
                # (e.g. tqdm). Forward the latest segment as a 'progress' line.
                if "\r" in buffer:
                    progress = buffer.rsplit("\r", 1)[-1].strip("\r")
                    if progress:
                        log_callback(progress, "progress")

            # Flush any remaining partial line at EOF
            if buffer.strip():
                log_callback(buffer.rstrip("\r"), stream_name)

        # Read both streams concurrently
        await asyncio.gather(
            read_stream(self.process.stdout, "stdout"),
            read_stream(self.process.stderr, "stderr"),
        )

    def _scan_output_files(self, output_dir: str) -> List[dict]:
        """Scan output directory for recently created files."""
        files = []
        output_path = Path(output_dir)
        if not output_path.exists():
            return files

        # Get files modified in the last hour
        import time
        cutoff = time.time() - 24 * 3600

        for f in output_path.rglob("*"):
            try:
                if f.is_file() and f.stat().st_mtime > cutoff:
                    files.append({
                        "name": f.name,
                        "path": str(f.absolute()),
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    })
            except (OSError, PermissionError):
                continue

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)
        return files[:50]

    def cancel(self):
        """Cancel the running process."""
        self._cancelled = True
        if self.process:
            try:
                if platform.system() == "Windows":
                    self.process.kill()
                else:
                    import signal
                    self.process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass


def open_folder(path: str):
    """Open a folder in the system file manager."""
    path = Path(path)
    if not path.exists():
        return

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)])
        elif system == "Windows":
            subprocess.run(["explorer", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception:
        pass


def open_file(path: str):
    """Open a file with the default application."""
    path = Path(path)
    if not path.exists():
        return

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)])
        elif system == "Windows":
            os.startfile(str(path))
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception:
        pass


def pick_directory(title: str = "Select Directory", initial: str = "") -> Optional[str]:
    """
    Open a native directory picker dialog (cross-platform via tkinter).
    Returns selected path or None if cancelled.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Bring dialog to front

    path = filedialog.askdirectory(
        title=title,
        initialdir=initial if initial and Path(initial).exists() else str(Path.home()),
    )

    root.destroy()
    return path if path else None


def pick_file(title: str = "Select File", filetypes: list = None, initial: str = "") -> Optional[str]:
    """
    Open a native file picker dialog (cross-platform via tkinter).
    Returns selected path or None if cancelled.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title=title,
        initialdir=initial if initial and Path(initial).exists() else str(Path.home()),
        filetypes=filetypes or [("All files", "*.*")],
    )

    root.destroy()
    return path if path else None
