
from pathlib import Path
import tempfile
import shutil

def get_runtime_dirs(session_state) -> dict:
    """Create per-session temp dirs under /tmp (or system temp)."""
    if "rt_dirs" not in session_state:
        base = Path(tempfile.mkdtemp(prefix="veh-plate-"))
        up = base / "uploaded"
        ex = base / "plates"
        up.mkdir(parents=True, exist_ok=True)
        ex.mkdir(parents=True, exist_ok=True)
        session_state["rt_dirs"] = {"base": base, "uploaded": up, "plates": ex}
    return session_state["rt_dirs"]

def cleanup_runtime_dirs(session_state):
    d = session_state.get("rt_dirs")
    if d:
        shutil.rmtree(d["base"], ignore_errors=True)
        session_state.pop("rt_dirs", None)

def models_root(app_file: Path) -> Path:
    """Read-only models live inside the repo and can be read just fine."""
    return app_file.parent / "models"
