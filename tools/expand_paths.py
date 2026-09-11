#!/usr/bin/env python3
"""Expand ${ROOT} in the OneTrainer configs and concept files.

OneTrainer reads the JSON directly and does no substitution, so the configs
ship with a token and get expanded in place before a run.

    python tools/expand_paths.py
    python tools/expand_paths.py --check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ROOT

TOK = "${ROOT}"


def walk(obj, hits):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and TOK in v:
                obj[k] = v.replace(TOK, str(ROOT))
                hits.append(obj[k])
            else:
                walk(v, hits)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    base = ROOT / "training"
    n = 0
    for f in sorted(base.rglob("*.json")):
        obj = json.loads(f.read_text())
        hits = []
        walk(obj, hits)
        if not hits:
            continue
        n += len(hits)
        print(f"{f.relative_to(ROOT)}: {len(hits)}")
        if not args.check:
            f.write_text(json.dumps(obj, indent=4))
    print(f"{n} paths {'found' if args.check else 'expanded'} -> {ROOT}")


if __name__ == "__main__":
    main()
