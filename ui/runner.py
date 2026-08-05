"""
DROCAT UI Runner Engine
Executes tools by generating proper Python scripts with user parameters.
"""

import asyncio
import os
import re
import sys
import tempfile
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any

from .config import PROJECT_ROOT, SRC_DIR


# The PlotPath tool lives in the vispath subproject, imported as `vispath_pkg`.
# It is not installed into site-packages, so generated scripts must add
# vispath-subproject/src to sys.path themselves.
VISPATH_DIR = PROJECT_ROOT / "vispath-subproject" / "src"


# =============================================================================
# Output stream sanitizer
# =============================================================================

# ANSI escape sequences: CSI (colors, cursor moves, line erases), OSC
# (title/link), and single-character escapes. Stripped so the UI log shows
# clean text instead of raw escape codes.
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-Z\\-_]"
)

# tqdm-style progress bar line, e.g.
# "Building target profiles:  45%|████▍       | 1328/2972 [00:05<00:06, 252.17it/s]"
_PROGRESS_LINE_RE = re.compile(r"^\s*.*?\d+%\|.*\|\s*\d+/\d+\s*\[.*\]\s*$")


class _OutputSplitter:
    """Incremental splitter turning raw subprocess output into clean log lines.

    Terminal-style output is normalized so the UI log stays readable:

    - ANSI escape sequences (colors, cursor moves, line erases) are stripped.
    - ``\r``-refreshed segments and cursor-up (``\x1b[A``) refreshes are
      yielded as *progress* items so the output panel can update one line
      in place instead of appending a new line per refresh.
    - Newline-terminated tqdm-style progress bars (non-TTY output) are also
      classified as progress.
    - Whitespace-only fragments (tqdm's line-clearing) and trailing spaces
      are dropped.

    ``feed`` consumes decoded chunks and yields ``(text, is_progress)`` pairs
    for every complete output segment; ``flush`` emits the final partial line.
    """

    def __init__(self):
        self._pending = ""

    def feed(self, chunk: str):
        self._pending += chunk
        # A cursor-up escape moves the cursor back to the previous progress
        # line; treat it like a carriage return (refresh separator).
        self._pending = self._pending.replace("\x1b[A", "\r")
        self._pending = _ANSI_ESCAPE_RE.sub("", self._pending)
        while True:
            nl = self._pending.find("\n")
            cr = self._pending.find("\r")
            if nl == -1 and cr == -1:
                break
            if nl != -1 and (cr == -1 or nl < cr):
                segment, self._pending = self._pending.split("\n", 1)
                refreshed = False
            else:
                segment, self._pending = self._pending.split("\r", 1)
                # A "\r\n" pair is a regular line ending (Windows), not an
                # in-place refresh; consume the "\n" so CRLF lines stay
                # ordinary log lines.
                if self._pending.startswith("\n"):
                    self._pending = self._pending[1:]
                    refreshed = False
                else:
                    refreshed = True
            yield from self._classify(segment, refreshed)

    def flush(self):
        if self._pending.strip():
            yield from self._classify(self._pending, False)
        self._pending = ""

    @staticmethod
    def _classify(segment, refreshed):
        text = segment.rstrip()
        if not text.strip():
            # tqdm line-clearing residue (spaces) or blank lines.
            return
        is_progress = refreshed or bool(_PROGRESS_LINE_RE.match(text))
        yield (text, is_progress)


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
            # Keep the UI constructor toggle when FindAllPath's method-level
            # default would otherwise overwrite it with False.
            "find_all_path": (
                "fc.FindAllPath(forward_only=True, "
                "find_reciprocal=fc.find_reciprocal)"
            ),
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
    """Format a Python value for code generation (repr round-trips exactly)."""
    return repr(value)


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
        self._run_logs: List[tuple] = []

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
        self._run_logs = []
        start_time = datetime.now()

        def _log(line: str, level: str = "stdout"):
            """Record every log line and forward it to the UI callback."""
            self._run_logs.append((level, line))
            if log_callback:
                log_callback(line, level)

        # Generate the script
        script_content = self._generate_script(
            tool_name, constructor_params, method_name, method_params
        )

        # Write to temp file
        fd, temp_script = tempfile.mkstemp(suffix=".py", prefix="drocat_run_")
        with os.fdopen(fd, "w") as f:
            f.write(script_content)

        try:
            label = tool.get("label", tool_name)
            _log("", "system")
            _log(f"━━━ ▶ UI FUNCTION: {label}  ({tool_name}) ━━━", "system")
            _log(
                "Output streams live below while the function runs; "
                "generated files appear in Output Files when it finishes.",
                "system",
            )
            _log("", "system")
            # Log params as multi-line for readability
            _log("[DROCAT] Parameters:", "system")
            for k, v in constructor_params.items():
                _log(f"  {k}: {v}", "system")
            if method_params:
                for k, v in method_params.items():
                    _log(f"  {k}: {v}", "system")

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
            await self._stream_output(_log)

            # Wait for completion
            returncode = await self.process.wait()

            duration = (datetime.now() - start_time).total_seconds()

            if self._cancelled:
                _log("[DROCAT] Execution cancelled by user.", "error")
            elif returncode == 0:
                _log(
                    f"━━━ ■ FINISHED: {tool.get('label', tool_name)} — "
                    f"completed in {duration:.1f}s ━━━",
                    "success",
                )
            else:
                _log(
                    f"━━━ ■ FINISHED: {tool.get('label', tool_name)} — "
                    f"failed with return code {returncode} ━━━",
                    "error",
                )

            # Scan only the folder THIS run actually generated. A run that
            # fails before creating its per-run folder must not fall back to
            # the shared storage root — that would list files from previous
            # runs (e.g. an earlier BANC run while running male-cns).
            scan_dir = self._resolve_scan_dir(output_dir)
            files = self._scan_output_files(scan_dir) if scan_dir else []

            return {
                "returncode": returncode,
                "files": files,
                "duration": duration,
                "cancelled": self._cancelled,
                "output_folder": scan_dir,
            }

        except Exception as e:
            _log(f"[DROCAT] Error: {str(e)}", "error")
            return {
                "returncode": -1,
                "files": [],
                "duration": 0,
                "cancelled": False,
                "output_folder": None,
            }
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
        elif tool_name == "plot3d_skeleton":
            return self._generate_plot3d_script(constructor_params, method_params)

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

    def _generate_plot3d_script(
        self, constructor_params: dict, method_params: Optional[dict]
    ) -> str:
        """Generate script for the 3D skeleton tool, including optional
        individual-profile PDF/PPTX export and rotating-video export."""
        params_str = _format_params(constructor_params)
        mp = method_params or {}

        extra_calls = ""
        if mp.get("export_individual_profiles"):
            extra_calls += f"""
# Export individual profiles (PDF/PPTX)
vs.plot_individuals(
    pdf_images_per_page={_format_value(mp.get('pdf_images_per_page', (3, 2)))},
    views={_format_value(mp.get('views', ['front']))},
    summary_format={_format_value(mp.get('summary_format', ['pdf']))},
)
"""
        if mp.get("export_video"):
            extra_calls += f"""
# Export rotating video / GIF
vs.export_video(
    fps={_format_value(mp.get('fps', 30))},
    degree_per_frame={_format_value(mp.get('degree_per_frame', 1.0))},
    rotate={_format_value(mp.get('rotate', 'horizontal'))},
    export_gif={_format_value(mp.get('export_gif', True))},
    gif_scale={_format_value(mp.get('gif_scale', 0.2))},
)
"""

        script = f'''#!/usr/bin/env python
"""Auto-generated DROCAT runner script for plot3d_skeleton."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, r"{SRC_DIR}")
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{VISPATH_DIR}")

warnings.filterwarnings("ignore")

from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
{params_str}
)

vs.plot_neurons()
{extra_calls}
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

        Reads chunks as soon as they arrive (unbuffered subprocess) and emits
        sanitized, complete lines immediately. tqdm-style progress bars and
        carriage-return refreshes are forwarded as 'progress' lines so the
        output panel updates one line in place instead of spamming a new line
        per update (see _OutputSplitter).
        """
        if not self.process:
            return

        async def read_stream(stream, stream_name):
            splitter = _OutputSplitter()
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                for text, is_progress in splitter.feed(
                    chunk.decode("utf-8", errors="replace")
                ):
                    log_callback(text, "progress" if is_progress else stream_name)
            for text, is_progress in splitter.flush():
                log_callback(text, "progress" if is_progress else stream_name)

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

    OUTPUT_FOLDER_MARKERS = [
        "Created output folder: ",
        "data will be saved in: ",
        "Output files in: ",
        "Output folder: ",
        "📁 Output folder: ",
        "📁 Output: ",
        "Saving results to: ",
        "Results saved to: ",
        "Output will be saved to: ",
    ]

    # Per-run output folder prefixes used by every DROCAT tool
    # (findpath_, findhomologs_, plotpath_, ...). A directory whose name does
    # not start with one of these is a shared storage root that may contain
    # many runs — it must never be scanned as the current run's folder.
    _RUN_FOLDER_PREFIX_RE = re.compile(
        r"^(findpath|findallpath|finddirect|findhomologs|profiling|interdataset|"
        r"plot3d|plotpath|colabel|findlines|findneuron)_"
    )

    def _resolve_scan_dir(self, output_dir: Optional[str] = None) -> Optional[str]:
        """Return the folder whose files belong to the current run.

        The per-run folder announced by the backend wins. Otherwise the
        caller-provided directory is used only when it is itself a per-run
        folder (plot_path pre-creates its own folder before the run). A
        shared storage root holding many runs is never scanned — its files
        belong to previous runs.
        """
        run_folder = self._extract_output_folder(output_dir)
        if run_folder:
            return run_folder
        if output_dir and self._RUN_FOLDER_PREFIX_RE.match(Path(output_dir).name):
            return output_dir
        return None

    def _extract_output_folder(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Return the folder the current run actually generated.

        Backend functions log where they create their per-run output folder
        (e.g. "Created output folder: ..."). The last logged path that exists
        and lies under the requested output directory wins, so the results
        panel links to the current run instead of older runs in the same
        storage directory.
        """
        candidates = []
        for _level, line in self._run_logs:
            for marker in self.OUTPUT_FOLDER_MARKERS:
                if marker in line:
                    candidate = line.split(marker, 1)[1].strip().strip("`").strip()
                    if candidate and os.path.isdir(candidate):
                        candidates.append(candidate)
                    break
        if not candidates:
            return None
        if output_dir:
            base = Path(output_dir).expanduser().resolve()
            # Prefer the newest top-level run folder (a direct child of the
            # requested output dir); avoid nested folders like
            # bodyId_visualization that internal tools also log.
            direct_children = [
                c for c in candidates
                if Path(c).expanduser().resolve().parent == base
            ]
            for candidate in reversed(direct_children or candidates):
                resolved = Path(candidate).expanduser().resolve()
                try:
                    resolved.relative_to(base)
                except ValueError:
                    continue
                # Return the logged spelling (important on macOS where
                # /var resolves to /private/var), after validating it safely.
                return candidate
            return None
        return candidates[-1]

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
