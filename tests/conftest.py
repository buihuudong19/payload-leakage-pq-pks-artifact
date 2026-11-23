import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Scripts started by the tests as subprocesses print UTF-8 text; on Windows a captured stdout would otherwise use the
# ANSI code page (cp1252) and fail on characters such as arrows or Greek letters. The orchestration scripts set the
# same variable (scripts/orchestration/common.ps1).
os.environ.setdefault("PYTHONUTF8", "1")


def load_bl1_reference_modules():
    """Load the BL1 reference implementation from baselines/bl1_reference (or PQPKS_BL1_REFERENCE_DIR).
    tkinter is replaced by a stub because the reference code calls messagebox during embedding."""
    tk = types.ModuleType("tkinter")
    mb = types.ModuleType("tkinter.messagebox")
    fd = types.ModuleType("tkinter.filedialog")
    mb.showerror = mb.showinfo = lambda *a, **k: None
    tk.messagebox, tk.filedialog = mb, fd
    sys.modules.setdefault("tkinter", tk)
    sys.modules.setdefault("tkinter.messagebox", mb)
    sys.modules.setdefault("tkinter.filedialog", fd)
    p = str(Path(os.environ.get("PQPKS_BL1_REFERENCE_DIR") or ROOT / "baselines" / "bl1_reference"))
    if not os.path.isfile(os.path.join(p, "Embed.py")):
        from pqpks.data import MissingResource
        raise MissingResource(f"BL1 reference code not found in {p} (see baselines/README.md)")
    if p not in sys.path:
        sys.path.append(p)
    import Embed  # noqa: E402
    import TextToBin  # noqa: E402
    return Embed, TextToBin


def pytest_runtest_makereport(item, call):
    """Report a test as skipped when it needs a resource that is not redistributed (BL1 reference code,
    the [Zha25] module, BOSSbase); see data/README.md and baselines/README.md."""
    import pytest
    from pqpks.data import MissingResource
    if call.excinfo is not None and call.excinfo.errisinstance(MissingResource):
        rep = pytest.TestReport.from_item_and_call(item, call)
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {call.excinfo.value}")
        return rep
