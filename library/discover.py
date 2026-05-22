#!/usr/bin/env python3
"""Discover library presets by scanning the judges/ and behaviors/ folders.

Usage:
    python library/discover.py              # list all presets
    python library/discover.py judges       # list judge presets only
    python library/discover.py behaviors    # list behavior presets only
    python library/discover.py --tags safety  # filter by tag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

LIBRARY_ROOT = Path(__file__).resolve().parent
SUBDIRS = {"judges": "judges", "behaviors": "behaviors"}


def discover(subdir: str | None = None, tags: list[str] | None = None) -> List[Dict[str, Any]]:
    """Scan YAML files and return metadata for each preset.

    Args:
        subdir: Optional subdirectory name ("judges" or "behaviors").
                If None, scans both.
        tags: Optional list of tags to filter by (any match).

    Returns:
        List of dicts with keys: name, kind, path, tags, description.
    """
    dirs = [subdir] if subdir else list(SUBDIRS.keys())
    results: List[Dict[str, Any]] = []
    for d in dirs:
        folder = LIBRARY_ROOT / SUBDIRS[d]
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob("*.yaml")):
            with open(f) as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict) or "name" not in data:
                continue
            entry = {
                "name": data["name"],
                "kind": data.get("kind", "unknown"),
                "path": f"{d}/{f.name}",
                "tags": data.get("tags", []),
                "description": data.get("description", "").strip(),
            }
            if tags and not set(tags) & set(entry["tags"]):
                continue
            results.append(entry)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="List library presets.")
    parser.add_argument("subdir", nargs="?", choices=["judges", "behaviors"],
                        help="Limit to a subdirectory")
    parser.add_argument("--tags", nargs="+", help="Filter by tags (any match)")
    args = parser.parse_args()

    entries = discover(subdir=args.subdir, tags=args.tags)
    if not entries:
        print("No presets found.")
        sys.exit(0)

    max_name = max(len(e["name"]) for e in entries)
    max_kind = max(len(e["kind"]) for e in entries)
    for e in entries:
        tags_str = ", ".join(e["tags"])
        print(f"  {e['name']:<{max_name}}  {e['kind']:<{max_kind}}  [{tags_str}]")


if __name__ == "__main__":
    main()
