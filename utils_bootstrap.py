from __future__ import annotations

from pathlib import Path
import tempfile
import urllib.request


def app_tmp_dir(*parts: str) -> str:
    """Return an app-specific temporary dir under the system temp, creating it.
    Example: app_tmp_dir("uploads") -> "/tmp/lpv_app/uploads" (Linux/Streamlit Cloud)
    """
    base = Path(tempfile.gettempdir()) / "lpv_app"
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_file(url: str, dst_path: str | Path) -> str:
    """Download a file if missing/empty. Useful for model weights.
    Returns the destination path as string.
    """
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size == 0:
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            f.write(r.read())
    return str(dst)
