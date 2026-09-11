#!/usr/bin/env python3
"""Check that every path in the archive points at a file which exists.

Covers configs to their concept and sample files, concepts to their dataset
dirs, file references in notebooks and scripts, the paths.py helpers, and that no Drive path
survives in a cell source. Paths in training/configs/missing.md
are reported as known gaps instead of failures.

Exits 1 on anything broken.

    python tools/check_refs.py
    python tools/check_refs.py -v
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOK = "${ROOT}"
REF = re.compile(r"\{ROOT\}/([\w/.()\- ]+\.(?:json|py|ipynb|txt|safetensors))")
DRIVE = re.compile(r"/content/drive/MyDrive/")
MODELS = ("sdxl", "pixart", "qwen", "flux2")

bad, gaps = [], []
n = 0


def known():
    f = ROOT / "training/configs/missing.md"
    return set(re.findall(r"`([^`]+)`", f.read_text())) if f.exists() else set()


def hit(v):
    return ROOT / v.replace(TOK, "").lstrip("/")


def main():
    global n
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    v = ap.parse_args().verbose
    skip = known()

    for f in sorted((ROOT / "training/configs").rglob("*.json")):
        cfg = json.loads(f.read_text())
        for k in ("concept_file_name", "sample_definition_file_name"):
            if not cfg.get(k):
                continue
            n += 1
            p = hit(cfg[k])
            rel = str(p.relative_to(ROOT))
            if p.exists():
                if v:
                    print(f"ok   {f.relative_to(ROOT)} {k} -> {rel}")
            elif rel in skip:
                gaps.append(f"{f.relative_to(ROOT)} {k} -> {rel}")
            else:
                bad.append(f"{f.relative_to(ROOT)} {k} -> {rel}")

    for f in sorted((ROOT / "training/concepts").glob("*/train_concepts_*.json")):
        for con in json.loads(f.read_text()):
            if not isinstance(con.get("path"), str) or not con["path"]:
                continue
            n += 1
            if not hit(con["path"]).is_dir():
                bad.append(f"{f.relative_to(ROOT)} path -> {con['path']}")

    seen = set()
    for f in sorted(list(ROOT.rglob("*.ipynb")) + list(ROOT.rglob("*.py"))):
        if f.name == "check_refs.py":
            continue
        if f.suffix == ".ipynb":
            src = ["".join(c.get("source", [])) for c in json.loads(f.read_text())["cells"]]
        else:
            src = [f.read_text()]
        for s in src:
            for m in REF.finditer(s):
                rel = m.group(1)
                key = (str(f.relative_to(ROOT)), rel)
                if key in seen:
                    continue
                seen.add(key)
                n += 1
                if (ROOT / rel).exists():
                    if v:
                        print(f"ok   {key[0]} -> {rel}")
                elif rel in skip:
                    gaps.append(f"{key[0]} -> {rel}")
                else:
                    bad.append(f"{key[0]} -> {rel}")

    sys.path.insert(0, str(ROOT))
    import paths
    for m in MODELS:
        for sp in paths.SPECIES:
            n += 1
            if not paths.config(m, sp).exists():
                bad.append(f"paths.config({m!r}, {sp!r})")

    for f in sorted(ROOT.rglob("*.ipynb")):
        for i, c in enumerate(json.loads(f.read_text())["cells"]):
            if DRIVE.search("".join(c.get("source", []))):
                bad.append(f"{f.relative_to(ROOT)} cell {i}: drive path")
    for f in sorted(ROOT.rglob("*.py")):
        if f.name != "check_refs.py" and DRIVE.search(f.read_text()):
            bad.append(f"{f.relative_to(ROOT)}: drive path")

    print(f"{n} refs checked")
    if gaps:
        print(f"{len(gaps)} known gaps (see training/configs/missing.md)")
        for g in gaps:
            print(f"  {g}")
    if bad:
        print(f"{len(bad)} broken")
        for b in bad:
            print(f"  {b}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
