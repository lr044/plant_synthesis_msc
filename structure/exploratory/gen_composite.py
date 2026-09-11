"""Composite the built plant onto a real habitat photo, then regenerate only
the plant. Exploratory, not used in the reported results.

The background is a real crop, the mask says regenerate inside the plant only,
so SDXL never gets to redraw the habitat.

Per image: pick a tile from the bank, paste the shaded plant over it with its
colour statistics pulled part way toward the tile's, ControlNet inpaint inside
the dilated mask, then an optional low-strength pass over the whole frame to
unify grain and light.

Needs, per spec, from build_plant.py: <name>.png line art, <name>_flat.png,
<name>_mask.png, <name>.txt and <name>.neg.txt.

    python gen_composite.py out/ --bg backgrounds -n 3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# torch is only needed once we actually generate
try:
    import torch
except ImportError:
    torch = None

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET = "xinsir/controlnet-canny-sdxl-1.0"
VAE = "madebyollin/sdxl-vae-fp16-fix"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

FALLBACK_PROMPT = ("a small clump of plants with pink flowers in a green "
                   "meadow, green leaves, realistic photograph, natural "
                   "daylight, sharp focus, fine leaf and petal detail, "
                   "shot on a 100mm macro lens at f/8")
FALLBACK_NEG = ("drawing, sketch, line art, illustration, cartoon, painting, "
                "flat colour, oversaturated, neon, lowres, worst quality, "
                "watermark, text, person, hands, floating plant, cut out, "
                "heavy bokeh, extreme background blur, soft focus")


def skeletonise(binary):
    try:
        return cv2.ximgproc.thinning(binary)
    except AttributeError:
        pass
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out, tmp = np.zeros_like(binary), binary.copy()
    while cv2.countNonZero(tmp):
        out = cv2.bitwise_or(out, cv2.subtract(
            tmp, cv2.morphologyEx(tmp, cv2.MORPH_OPEN, k)))
        tmp = cv2.erode(tmp, k)
    return out


def colour_transfer(src, ref, mask, amount=0.55):
    """
    Pull the plant's LUMINANCE toward the habitat tile's, and its chroma only
    slightly.

    What reads as a paste is a mismatch in brightness and contrast, not hue. An
    even LAB match across all three channels drags the flowers toward the
    background's green and washes them out, which destroys the one attribute
    the whole pipeline exists to control. So L is matched at full strength and
    a/b at a quarter of it.
    """
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask > 127
    if m.sum() < 64:
        return src
    out = s.copy()
    for c, w in enumerate((1.0, 0.25, 0.25)):
        a = amount * w
        if a <= 0:
            continue
        sm, ss = s[..., c][m].mean(), s[..., c][m].std() + 1e-6
        rm, rs = r[..., c].mean(), r[..., c].std() + 1e-6
        adj = (s[..., c] - sm) * (rs / ss) + rm
        out[..., c] = s[..., c] * (1 - a) + adj * a
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def build_composite(flat_path, mask_path, bg_path, size, feather=3,
                    transfer=0.55, bg_blur=1.6, foreground=0.45, shadow=0.35,
                    rng=None):
    """Plant pasted onto the habitat tile, plus the mask used to do it."""
    flat = cv2.imread(str(flat_path))
    mask = cv2.imread(str(mask_path), 0)
    bg = cv2.imread(str(bg_path))
    if flat is None or mask is None or bg is None:
        raise SystemExit(f"cannot read one of {flat_path} {mask_path} {bg_path}")

    W = H = size
    flat = cv2.resize(flat, (W, H), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # random crop of the tile rather than a squash, so the bank goes further
    bh, bw = bg.shape[:2]
    side = min(bh, bw)
    if rng is not None and side > size:
        s2 = rng.randint(int(side * 0.72), side)
        x = rng.randint(0, bw - s2)
        y = rng.randint(0, bh - s2)
        bg = bg[y:y + s2, x:x + s2]
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
    if bg_blur > 0:
        # tiles are close crops, so their vegetation sits at the same apparent
        # scale as the subject. A little blur pushes it behind the plant and
        # stands in for the depth of field a real photograph would have.
        bg = cv2.GaussianBlur(bg, (0, 0), bg_blur)

    plant = colour_transfer(flat, bg, mask, transfer)
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather)
    a = np.clip(a, 0, 1)[:, :, None]

    # contact shadow: darken the habitat just below and beside the plant, so
    # the stems sit on the ground rather than hovering over it
    if shadow > 0:
        sh = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), H * 0.02)
        M = np.float32([[1, 0, W * 0.012], [0, 0.28, H * 0.72]])
        sh = cv2.warpAffine(sh, M, (W, H))
        sh = np.clip(sh, 0, 1)[:, :, None]
        bg = np.clip(bg.astype(np.float32) * (1 - shadow * sh), 0, 255
                     ).astype(np.uint8)

    comp = (plant.astype(np.float32) * a + bg.astype(np.float32) * (1 - a))

    # foreground occlusion: paint the tile's own vegetation back over the
    # bottom of the plant. A real plant in a meadow has grass crossing in front
    # of its base; a pasted one never does, and that is what makes the paste
    # obvious no matter how well the colours match.
    keep = mask.copy()
    if foreground > 0:
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        hh = hsv[..., 0].astype(np.float32)
        ss, vv = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
        veg = (((hh >= 30) & (hh <= 95)) & (ss > 0.20) & (vv > 0.12)
               ).astype(np.float32)
        band = np.clip((np.arange(H) - H * (1 - foreground))
                       / max(H * foreground, 1), 0, 1)[:, None] ** 1.4
        fg = cv2.GaussianBlur(veg * band, (0, 0), 2.0)
        fg = np.clip(fg * 1.25, 0, 1)
        comp = comp * (1 - fg[:, :, None]) + bg.astype(np.float32) * fg[:, :, None]

        keep = np.clip(mask.astype(np.float32) * (1 - fg), 0, 255).astype(np.uint8)

    return comp.astype(np.uint8), keep


def prep_control(line_path, size):
    img = cv2.imread(str(line_path))
    g = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    if g.mean() > 127:
        g = 255 - g

    edges = skeletonise(cv2.threshold(g, 60, 255, cv2.THRESH_BINARY)[1])
    return Image.fromarray(np.stack([edges] * 3, -1))


def find_specs(inputs):
    out = []
    for p in inputs:
        p = Path(p)
        cands = sorted(p.glob("*.png")) if p.is_dir() else [p]
        for f in cands:
            n = f.stem
            if n.endswith(("_flat", "_mask", "_control", "_composite")):
                continue
            if n[-5:-4] == "_" and n[-4:].isdigit():
                continue
            if f.with_name(n + "_mask.png").exists():
                out.append(f)
    if not out:
        raise SystemExit("no specs found (need <name>.png with <name>_mask.png)")
    return out


def sidecar(path, suffix, fallback):
    f = path.with_suffix(suffix)
    return f.read_text().strip() if f.exists() else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--bg", default="backgrounds",
                    help="background bank from extract_backgrounds.py")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=0.88,
                    help="inpaint strength inside the plant mask. 1.0 ignores "
                         "the composited colour entirely")
    ap.add_argument("--scale", type=float, default=0.75,
                    help="controlnet_conditioning_scale")
    ap.add_argument("--control-end", type=float, default=0.8)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dilate", type=int, default=12,
                    help="mask dilation in px; gives the model room to blend "
                         "the plant edge into the habitat")
    ap.add_argument("--transfer", type=float, default=0.55,
                    help="how far to pull plant luminance toward the tile, "
                         "0-1; chroma moves at a quarter of this")
    ap.add_argument("--foreground", type=float, default=0.45,
                    help="fraction of frame height over which the tile's own "
                         "vegetation is painted back in front of the plant. "
                         "This is what stops it looking pasted on")
    ap.add_argument("--shadow", type=float, default=0.35,
                    help="contact shadow strength under the plant")
    ap.add_argument("--bg-blur", type=float, default=1.6,
                    help="sigma of background blur; separates the subject and "
                         "stands in for depth of field. 0 disables")
    ap.add_argument("--unify", type=float, default=0.15,
                    help="final whole-frame img2img strength to match grain "
                         "and light; 0 disables")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-composite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="write composites only, load no models")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tiles = sorted(p for p in (Path(args.bg) / "tiles").iterdir()
                   if p.suffix.lower() in EXTS) if (
        Path(args.bg) / "tiles").is_dir() else sorted(
        p for p in Path(args.bg).iterdir() if p.suffix.lower() in EXTS)
    if not tiles:
        raise SystemExit(f"no background tiles under {args.bg}")
    specs = find_specs(args.inputs)
    print(f"{len(specs)} spec(s), {len(tiles)} background tiles, "
          f"{args.n} image(s) each")

    pipe = unify_pipe = None
    if not args.dry_run:
        if torch is None:
            raise SystemExit("torch is not installed; use --dry-run to write "
                             "composites only")
        from diffusers import (AutoencoderKL, ControlNetModel,
                               EulerAncestralDiscreteScheduler,
                               StableDiffusionXLControlNetInpaintPipeline,
                               StableDiffusionXLImg2ImgPipeline)
        common = dict(
            vae=AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16),
            torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            BASE,
            controlnet=ControlNetModel.from_pretrained(
                CONTROLNET, torch_dtype=torch.float16),
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
                BASE, subfolder="scheduler"),
            **common).to("cuda")
        pipe.set_progress_bar_config(leave=False)
        if args.unify > 0:
            unify_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                BASE, **common).to("cuda")
            unify_pipe.set_progress_bar_config(leave=False)

    for f in specs:
        out_dir = Path(args.out) if args.out else f.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = sidecar(f, ".txt", FALLBACK_PROMPT)
        negative = sidecar(f, ".neg.txt", FALLBACK_NEG)
        ctrl = prep_control(f, args.size)
        print(f"\n{f.name}")

        for i in range(args.n):
            rng = random.Random(args.seed + i + hash(f.stem) % 9973)
            tile = rng.choice(tiles)
            comp, mask = build_composite(
                f.with_name(f.stem + "_flat.png"),
                f.with_name(f.stem + "_mask.png"),
                tile, args.size, transfer=args.transfer,
                bg_blur=args.bg_blur, foreground=args.foreground,
                shadow=args.shadow, rng=rng)

            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (args.dilate * 2 + 1,) * 2)
            infill = cv2.dilate(mask, k)
            comp_pil = Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(infill)

            if args.save_composite or args.dry_run:
                comp_pil.save(out_dir / f"{f.stem}_composite_{i}.png")
            if args.dry_run:
                print(f"    composite {i}  bg={tile.name}")
                continue

            gen = torch.Generator("cuda").manual_seed(args.seed + i)
            img = pipe(prompt=prompt, negative_prompt=negative,
                       image=comp_pil, mask_image=mask_pil, control_image=ctrl,
                       strength=args.strength,
                       controlnet_conditioning_scale=args.scale,
                       control_guidance_start=0.0,
                       control_guidance_end=args.control_end,
                       guidance_scale=args.guidance,
                       num_inference_steps=args.steps,
                       width=args.size, height=args.size,
                       generator=gen).images[0]

            if unify_pipe is not None:
                img = unify_pipe(prompt=prompt, negative_prompt=negative,
                                 image=img, strength=args.unify,
                                 guidance_scale=args.guidance,
                                 num_inference_steps=max(args.steps // 2, 12),
                                 generator=gen).images[0]

            p = out_dir / f"{f.stem}_{args.seed + i}.png"
            img.save(p)
            print(f"    {p.name}  bg={tile.name}")

    print("\ndone")


if __name__ == "__main__":
    main()
