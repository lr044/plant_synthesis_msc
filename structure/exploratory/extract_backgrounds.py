"""Background bank from the real photographs. Exploratory, not used in the
reported results.

Procedural backgrounds smoothed, so this attempts to use real backgrounds. The method was unsuccessful

The dataset images are cropped around a focal plant so the subject is central.
Tiles are sampled away from the centre and scored on green fraction, white
fraction (a white petal is bright and unsaturated, so the saturated-hue test
misses it), largest saturated non-green blob, edge density, Laplacian variance
and exposure. Thresholds were fitted against 48 hand-labelled tiles. Recall is
sacrificed on purpose: a rejected good tile costs nothing, an accepted bad one
puts a second species in every background.

    python extract_backgrounds.py --images-dir images --out-dir backgrounds \
        --target 2000 --size 768
    python extract_backgrounds.py --images-dir images --contact-sheet 48
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# OpenCV hue is 0-179; vegetation sits roughly 30-90 (yellow-green to cyan)
GREEN_LO, GREEN_HI = 33, 92


def score_tile(bgr):
    """
    Return (ok, metrics). Thresholds were fitted against 48 hand-labelled tiles
    from a first run, where roughly a third were usable. The failure modes that
    mattered, and what catches each:

      herbarium sheets      green fraction. The dataset contains specimen scans
                            with paper and handwriting.
      white flowers         a white petal is bright and UNSATURATED, so the
                            saturated-hue test misses it entirely. Needs its own
                            term, and it was the single largest miss.
      dominant flowers      largest connected saturated non-green blob, rather
                            than total area: a flower strip along one edge is
                            only ~2% of a tile and passes any area threshold.
      macro leaf close-ups  edge density. A single leaf filling the frame has
                            few long edges; habitat vegetation has many short
                            ones. Wrong scale for a background.
      motion blur / bokeh   Laplacian variance.

    Everything is measured at a fixed 256px so thresholds do not shift with
    tile size. On the labelled set this keeps 10 of 19 good tiles and 1 of 29
    bad ones. Recall is deliberately sacrificed: there are 9,600 source images
    and only ~2,000 tiles are needed, so a rejected good tile costs nothing
    while an accepted bad one puts a second species in every background.
    """
    b = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32)
    s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    green = float((((h >= GREEN_LO) & (h <= GREEN_HI))
                   & (s > 0.18) & (v > 0.10)).mean())
    white = float(((v > 0.72) & (s < 0.28)).mean())
    yellow = float(((h >= 15) & (h < GREEN_LO) & (s > 0.35) & (v > 0.45)).mean())

    ng = (((h < GREEN_LO) | (h > GREEN_HI)) & (s > 0.30) & (v > 0.22)
          ).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(ng, 8)
    blob = float(st[1:, cv2.CC_STAT_AREA].max() / ng.size) if n > 1 else 0.0

    edge = float((cv2.Canny(gray, 60, 150) > 0).mean())
    detail = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    mean_v = float(v.mean())

    ok = (green >= 0.55 and white <= 0.10 and yellow <= 0.08 and blob <= 0.008
          and 0.15 <= edge <= 0.45 and detail >= 150.0
          and 0.14 < mean_v < 0.88)
    return ok, dict(green=green, white=white, yellow=yellow, blob=blob,
                    edge=edge, detail=detail, mean_v=mean_v)


def flower_exclusion(img, dilate_frac=0.04):
    """
    Where the focal plant's flowers are, dilated generously.

    Per-tile statistics cannot catch this: a flower strip along one edge of a
    tile is only ~2% of its pixels and passes every area threshold. Finding the
    flowers once on the whole image and then refusing to sample near them uses
    the global context that per-tile scoring discards.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    m = (((h < GREEN_LO) | (h > GREEN_HI)) & (s > 0.30) & (v > 0.22)
         ).astype(np.uint8)
    k = max(3, int(min(img.shape[:2]) * 0.012) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
   
    min_area = max(64, int(img.shape[0] * img.shape[1] * 2e-3))
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    d = max(3, int(min(img.shape[:2]) * dilate_frac) | 1)
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))


def sample_tiles(img, n_try, tile_frac, centre_avoid, rng, exclude=None):
    """Candidate crops away from the centre, where the focal plant sits."""
    H, W = img.shape[:2]
    t = int(min(H, W) * tile_frac)
    if t < 96:
        return []
    cx, cy = W / 2, H / 2
    rx, ry = W * centre_avoid, H * centre_avoid
    out = []
    for _ in range(n_try):
        x = rng.randint(0, max(W - t, 0))
        y = rng.randint(0, max(H - t, 0))
        mx, my = x + t / 2, y + t / 2
        # skip if the tile centre falls inside the subject ellipse
        if ((mx - cx) / max(rx, 1)) ** 2 + ((my - cy) / max(ry, 1)) ** 2 < 1.0:
            continue
        if exclude is not None and exclude[y:y + t, x:x + t].mean() > 0.01:
            continue
        out.append((x, y, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", default="backgrounds")
    ap.add_argument("--size", type=int, default=768,
                    help="output tile size; tiles are upscaled to this")
    ap.add_argument("--tile-frac", type=float, default=0.46,
                    help="crop side as a fraction of the image's short side")
    ap.add_argument("--centre-avoid", type=float, default=0.30,
                    help="half-axes of the central ellipse to avoid, as a "
                         "fraction of the image; the focal plant lives there")
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--tries", type=int, default=40,
                    help="candidates per image. The filter is "
                         "deliberately strict, so sample hard")
    ap.add_argument("--flower-dilate", type=float, default=0.04,
                    help="keep-out radius around detected "
                         "flowers, as a fraction of the short side")
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--min-source", type=int, default=380,
                    help="skip images whose short side is below this; small "
                         "sources upscale into mush")
    ap.add_argument("--contact-sheet", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    (out / "tiles").mkdir(parents=True, exist_ok=True)

    files = [p for p in Path(args.images_dir).rglob("*")
             if p.suffix.lower() in EXTS]
    rng.shuffle(files)
    print(f"{len(files)} source images")

    manifest, kept, seen, skipped_small = [], 0, 0, 0
    for f in files:
        if kept >= args.target:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        seen += 1
        if min(img.shape[:2]) < args.min_source:
            skipped_small += 1
            continue

        excl = flower_exclusion(img, args.flower_dilate)
        cands = sample_tiles(img, args.tries, args.tile_frac,
                             args.centre_avoid, rng, excl)
        scored = []
        for x, y, t in cands:
            tile = img[y:y + t, x:x + t]
            ok, m = score_tile(tile)
            if ok:
                # rank by edge density: prefererence the busiest vegetation,
                # which is what the real habitat photographs look like
                scored.append((m["edge"], x, y, t, m))
        scored.sort(reverse=True)

        for _rank, x, y, t, m in scored[: args.per_image]:
            tile = cv2.resize(img[y:y + t, x:x + t], (args.size, args.size),
                              interpolation=cv2.INTER_LANCZOS4)
            name = f"bg_{kept:05d}.jpg"
            cv2.imwrite(str(out / "tiles" / name), tile,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append(dict(tile=name, source=str(f), x=x, y=y, size=t,
                                 species=f.parent.name,
                                 **{k: round(v, 4) for k, v in m.items()}))
            kept += 1
            if kept >= args.target:
                break
        if seen % 500 == 0:
            print(f"  scanned {seen}, kept {kept}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{kept} tiles from {len({m['source'] for m in manifest})} images, "
          f"{len({m['species'] for m in manifest})} species")
    if skipped_small:
        print(f"{skipped_small} images skipped as too small "
              f"(< {args.min_source}px short side)")
    print(f"-> {out / 'tiles'}")

    if args.contact_sheet and manifest:
        n = min(args.contact_sheet, len(manifest))
        cols = 8
        rows = (n + cols - 1) // cols
        th = 200
        sheet = np.full((rows * th, cols * th, 3), 255, np.uint8)
        for i, m in enumerate(rng.sample(manifest, n)):
            t = cv2.imread(str(out / "tiles" / m["tile"]))
            t = cv2.resize(t, (th, th))
            r, c = divmod(i, cols)
            sheet[r * th:(r + 1) * th, c * th:(c + 1) * th] = t
        cv2.imwrite(str(out / "contact_sheet.jpg"), sheet)
        print(f"contact sheet -> {out / 'contact_sheet.jpg'}")
        print("check the sheet. a tile with the focal plant in it puts a "
              "second species in every background.")


if __name__ == "__main__":
    main()
