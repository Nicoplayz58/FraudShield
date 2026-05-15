from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "src",
        script_dir.parent.parent / "src",
        Path.cwd() / "databricks" / "src",
        Path.cwd() / "src",
    ]

    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    raise ModuleNotFoundError(
        "No se encontro la carpeta src de Databricks. Verifica que databricks/src exista en el workspace o en el repo sincronizado."
    )