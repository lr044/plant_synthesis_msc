"""Line art plus flat colour map -> photograph.

Three channels:
  controlnet (canny)  <- <name>.png        shape, margin, topology
  img2img init        <- <name>_flat.png   colour, bound to position
  prompt              <- <name>.txt        texture and style only

Prompt-only colour does not bind. "pink flowers" says what colour exists, not
where, so it bled onto the foliage. The flat map puts the colour at the right
coordinates instead.

strength is the dial: how much of the flat map SDXL may overwrite. 0.55 holds
placement tightly and can stay flat, 0.70 is the default, 0.85 is more photographic but placement drifts, 1.0 ignores the initialisation entirely.

    python build_plant.py --sweep leaf_margin -o out/m
    python gen_flat.py out/ -n 3
    python gen_flat.py out/m_serrated.png --strength 0.6 --scale 0.8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET = "xinsir/controlnet-canny-sdxl-1.0"
VAE = "madebyollin/sdxl-vae-fp16-fix"

FALLBACK_PROMPT = (
    "a single plant with pink flowers, green leaves, "
    "leaves densely covered in fine hairs, "
    "growing against a plain out-of-focus background, "
    "realistic close-up photograph, natural daylight, deep depth of field, "
    "everything in sharp focus, high detail"
)
FALLBACK_NEG = (
    "drawing, sketch, line art, illustration, cartoon, painting, flat colour, "
    "oversaturated, neon, blurry, lowres, worst quality, watermark, text, "
    "person, human, face, hands"
)


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


def prep(line_path, flat_path, res):
    img = cv2.imread(str(line_path))
    if img is None:
        raise SystemExit(f"cannot read {line_path}")
    h, w = img.shape[:2]
    r = np.sqrt(res * res / (w * h))
    nw = max(64, int(round(w * r / 64)) * 64)
    nh = max(64, int(round(h * r / 64)) * 64)

    g = cv2.cvtColor(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    if g.mean() > 127:
        g = 255 - g
    # the input is already a drawing, so thin the strokes rather than running
    # edge detection on them (Canny on a line turns it into two parallel lines)
    edges = skeletonise(cv2.threshold(g, 60, 255, cv2.THRESH_BINARY)[1])
    ctrl = Image.fromarray(np.stack([edges] * 3, -1))

    if flat_path and Path(flat_path).exists():
        init = Image.open(flat_path).convert("RGB").resize(
            (nw, nh), Image.LANCZOS)
    else:
        init = None
    return ctrl, init, nw, nh


def find_specs(inputs):
    """Each spec is a <name>.png with an optional <name>_flat.png beside it."""
    out = []
    for p in inputs:
        p = Path(p)
        cands = (sorted(p.glob("*.png")) if p.is_dir() else [p])
        for f in cands:
            n = f.stem
            if n.endswith("_flat") or n.endswith("_control") or n[-5:-4] == "_" \
                    and n[-4:].isdigit():
                continue
            out.append(f)
    if not out:
        raise SystemExit("no line-art PNGs found")
    return out


def sidecar(path, suffix, fallback):
    f = path.with_suffix(suffix)
    return f.read_text().strip() if f.exists() else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--strength", type=float, default=0.70,
                    help="img2img denoising strength; how much of the flat "
                         "colour map SDXL may overwrite")
    ap.add_argument("--scale", type=float, default=0.75,
                    help="controlnet_conditioning_scale")
    ap.add_argument("--control-end", type=float, default=0.75)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--no-flat", action="store_true",
                    help="ignore the flat map; ControlNet + prompt only")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None,
                    help="defaults to alongside each input")
    args = ap.parse_args()

    specs = find_specs(args.inputs)
    print(f"{len(specs)} spec(s), {args.n} image(s) each")

    from diffusers import (AutoencoderKL, ControlNetModel,
                           EulerAncestralDiscreteScheduler,
                           StableDiffusionXLControlNetImg2ImgPipeline,
                           StableDiffusionXLControlNetPipeline)

    common = dict(
        controlnet=ControlNetModel.from_pretrained(CONTROLNET,
                                                   torch_dtype=torch.float16),
        vae=AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16),
        scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
            BASE, subfolder="scheduler"),
        torch_dtype=torch.float16,
    )
    pipe_i2i = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        BASE, **common).to("cuda")
    pipe_t2i = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE, **common).to("cuda")
    for p in (pipe_i2i, pipe_t2i):
        p.set_progress_bar_config(leave=False)

    for f in specs:
        flat = None if args.no_flat else f.with_name(f.stem + "_flat.png")
        ctrl, init, w, h = prep(f, flat, args.resolution)
        prompt = sidecar(f, ".txt", FALLBACK_PROMPT)
        negative = sidecar(f, ".neg.txt", FALLBACK_NEG)
        out_dir = Path(args.out) if args.out else f.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        ctrl.save(out_dir / f"{f.stem}_control.png")

        mode = "controlnet+img2img" if init is not None else "controlnet only"
        print(f"\n{f.name}  {w}x{h}  [{mode}]")
        print(f"  {prompt[:98]}...")

        for i in range(args.n):
            gen = torch.Generator("cuda").manual_seed(args.seed + i)
            kw = dict(prompt=prompt, negative_prompt=negative,
                      controlnet_conditioning_scale=args.scale,
                      control_guidance_start=0.0,
                      control_guidance_end=args.control_end,
                      guidance_scale=args.guidance,
                      num_inference_steps=args.steps, generator=gen)
            if init is not None:
                im = pipe_i2i(image=init, control_image=ctrl,
                              strength=args.strength, **kw).images[0]
            else:
                im = pipe_t2i(image=ctrl, width=w, height=h, **kw).images[0]
            p = out_dir / f"{f.stem}_{args.seed + i}.png"
            im.save(p)
            print(f"    {p}")

    print("\ndone")


if __name__ == "__main__":
    main()
