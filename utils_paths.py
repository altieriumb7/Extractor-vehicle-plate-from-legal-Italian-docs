# utils_paths.py
from pathlib import Path
import tempfile
import shutil

import os, tempfile, urllib.request

def ensure_file(url: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
        with urllib.request.urlopen(url) as r, open(dst_path, "wb") as f:
            f.write(r.read())
    return dst_path

def app_tmp_dir(*parts: str) -> str:
    base = os.path.join(tempfile.gettempdir(), "lpv_app")
    p = os.path.join(base, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def get_runtime_dirs(session_state) -> dict:
    """
    Create per-session temp dirs under the system temp folder.
    Returns a dict with keys: base, uploaded, plates.
    """
    if "rt_dirs" not in session_state:
        base = Path(tempfile.mkdtemp(prefix="veh-plate-"))
        up = base / "uploaded"
        ex = base / "plates"
        up.mkdir(parents=True, exist_ok=True)
        ex.mkdir(parents=True, exist_ok=True)
        session_state["rt_dirs"] = {"base": base, "uploaded": up, "plates": ex}
    return session_state["rt_dirs"]

def cleanup_runtime_dirs(session_state):
    """Remove the whole per-session workspace."""
    d = session_state.get("rt_dirs")
    if d:
        shutil.rmtree(d["base"], ignore_errors=True)
        session_state.pop("rt_dirs", None)

def models_root(app_file: Path) -> Path:
    """
    Read-only models live inside the repo and can be read just fine.
    Example expected layout:
      models/
        yolo/best.pt
        paddleOCR/...
    """
    return app_file.parent / "models"
