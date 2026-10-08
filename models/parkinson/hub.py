"""Skore Hub credentials and project connection.

Usage
-----
    from parkinson.hub import load_skore_credentials, get_project

    cfg = load_skore_credentials()   # sets env-vars, returns public fields
    project = get_project(cfg)       # ready-to-use skore.Project
    project.put("01_dummy", report)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from skore import Project, login


def _find_skore(path: Path | None = None) -> Path:
    """Return the nearest ``.skore``, walking up from CWD or ``path``."""
    if path is not None:
        dest = Path(path)
        if dest.is_file():
            return dest
        raise FileNotFoundError(f"Missing {dest}. Run: python scripts/skore-agent")

    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in starts:
        for folder in [start, *start.parents]:
            if folder in seen:
                continue
            seen.add(folder)
            candidate = folder / ".skore"
            if candidate.is_file():
                return candidate
            if (folder / ".git").exists():
                break
    raise FileNotFoundError("Missing .skore. Run: python scripts/skore-agent")


def load_skore_credentials(path: Path | None = None) -> dict:
    """Read the nearest ``.skore`` and export hub URI + API key into ``os.environ``.

    Parameters
    ----------
    path:
        Optional explicit path to ``.skore``. When omitted, walks up from CWD.

    Returns
    -------
    dict
        Public fields: ``hub_url``, ``workspace``, ``workspace_id``.
        The API key is only written to ``os.environ``, never returned.
    """
    dest = _find_skore(path)
    try:
        cfg = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{dest} is not valid JSON") from exc
    if not isinstance(cfg, dict) or not cfg.get("hub_url") or not cfg.get("api_key"):
        raise ValueError(f"{dest} is missing hub_url or api_key. Run: python scripts/skore-agent")
    os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    return {k: v for k, v in cfg.items() if k != "api_key"}


def get_project(cfg: dict | None = None) -> Project:
    """Return a connected ``skore.Project`` on Hub.

    Calls ``load_skore_credentials`` when ``cfg`` is not already provided,
    then ``login(mode="hub")``.

    Parameters
    ----------
    cfg:
        Dict previously returned by ``load_skore_credentials()``.
        When ``None``, credentials are loaded automatically.
    """
    if cfg is None:
        cfg = load_skore_credentials()
    login(mode="hub")
    return Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
