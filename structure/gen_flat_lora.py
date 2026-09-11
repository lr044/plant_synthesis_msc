"""gen_flat.py with a per-species LoRA merged in.

Same three channels as gen_flat.py. With --lora the prompt should name the
rare token rather than describing the species.

The LoRA and ControlNet fight over the same image: raising scale suppresses
the LoRA on surface texture, raising lora-multiplier pulls the image off the drawn geometry. Sweep both before committing to a set.

    python gen_flat.py out/carpobrotus \
        --lora outputs/lora/sdxl_carpobrotus/lora.safetensors \
        --prompt "a detailed realistic photograph of z47mfpwjhs growing in sand dunes" \
        --suffix _lora -n 1
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


# Sidecar and derived suffixes that are never themselves a spec. `_mask` was
# missing, so masks were being treated as line art and generated against.
SKIP_SUFFIXES = ("_flat", "_control", "_mask")


def find_specs(inputs, recursive=True):
    """Each spec is a <name>.png with an optional <name>_flat.png beside it.

    Directories are searched recursively by default, because
    `build_species.py --species all` writes into one subdirectory per species
    and a flat glob finds nothing at the top level.
    """
    out = []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            cands = sorted(p.rglob("*.png") if recursive else p.glob("*.png"))
        else:
            cands = [p]
        for f in cands:
            n = f.stem
            if n.endswith(SKIP_SUFFIXES):
                continue
            # generated outputs carry a _<4-digit seed> suffix
            if n[-5:-4] == "_" and n[-4:].isdigit():
                continue
            out.append(f)
    if not out:
        raise SystemExit("no line-art PNGs found")
    return out


def sidecar(path, suffix, fallback):
    f = path.with_suffix(suffix)
    return f.read_text().strip() if f.exists() else fallback


# ---------------------------------------------------------------- LoRA merge
#
# OneTrainer emits two key conventions depending on the model:
#   flattened:  lora_unet_add_embedding_linear_1.lora_down.weight
#   dotted:     unet.add_embedding.linear_1.lora_down.weight
# The underscore flattening is not reversible, so rather than parsing keys we walk the real module
#  tree and construct the key each module would have under
# both conventions, keeping whichever matches.

ROOT_ATTRS = ["unet", "text_encoder", "text_encoder_2"]
KOHYA_ALIAS = {"text_encoder": "te1", "text_encoder_2": "te2"}


def _candidate_maps(root, attr):
    mods = [(n, m) for n, m in root.named_modules()
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))]
    alias = KOHYA_ALIAS.get(attr, attr)
    flat = f"lora_{alias}_"
    yield flat, {flat + n.replace(".", "_"): m for n, m in mods}
    dotted = f"{attr}."
    yield dotted, {dotted + n: m for n, m in mods}


def merge_lora(pipe, lora_path, multiplier=1.0, verbose=True):
    """Merge a LoRA into the pipeline in place. Irreversible."""
    from safetensors.torch import load_file

    sd = load_file(lora_path)
    prefixes = {k[: -len(".lora_down.weight")]
                for k in sd if k.endswith(".lora_down.weight")}
    merged, matched, seen, schemes = 0, set(), set(), []

    for attr in ROOT_ATTRS:
        root = getattr(pipe, attr, None)
        if root is None:
            continue
        for label, cand in _candidate_maps(root, attr):
            hits = set(cand) & prefixes
            if not hits:
                continue
            schemes.append(f"{label}*{len(hits)}")
            for key in hits:
                module = cand[key]
                if id(module) in seen:
                    continue
                down = sd[f"{key}.lora_down.weight"].to(torch.float32)
                up = sd[f"{key}.lora_up.weight"].to(torch.float32)
                rank = down.shape[0]
                akey = f"{key}.alpha"
                alpha = float(sd[akey]) if akey in sd else float(rank)
                scale = (alpha / rank) * multiplier
                w = module.weight
                try:
                    delta = (up.reshape(-1, rank)
                             @ down.reshape(rank, -1)).reshape(w.shape)
                except RuntimeError as e:
                    if verbose:
                        print(f"    skip {key}: {e}")
                    continue
                with torch.no_grad():
                    w.add_((delta * scale).to(w.dtype).to(w.device))
                merged += 1
                matched.add(key)
                seen.add(id(module))

    if verbose:
        print(f"  LoRA merged {merged}/{len(prefixes)} modules via "
              f"{', '.join(schemes) or 'nothing'} (multiplier {multiplier})")
        if merged == 0:
            raise SystemExit(
                "no LoRA modules matched the pipeline; check the file is an "
                "SDXL LoRA and not one trained on another backbone")
        if merged < len(prefixes) * 0.5:
            print("  less than half the modules matched")
    return merged, len(prefixes)


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
    ap.add_argument("--no-recursive", action="store_true",
                    help="do not descend into subdirectories")
    ap.add_argument("--lora", default=None,
                    help="a .safetensors LoRA to merge into the pipeline "
                         "before generating")
    ap.add_argument("--lora-multiplier", type=float, default=1.0,
                    help="LoRA strength; lower it when ControlNet and the "
                         "LoRA fight over the same image")
    ap.add_argument("--prompt", default=None,
                    help="override the per-spec .txt sidecar. With --lora "
                         "this should name the rare token rather than "
                         "describing the species")
    ap.add_argument("--negative", default=None,
                    help="override the per-spec .neg.txt sidecar")
    ap.add_argument("--suffix", default="",
                    help="extra suffix on output filenames, so a LoRA run "
                         "does not overwrite a base run")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None,
                    help="defaults to alongside each input")
    args = ap.parse_args()

    specs = find_specs(args.inputs, recursive=not args.no_recursive)
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

    if args.lora and not Path(args.lora).exists():
        raise SystemExit(f"LoRA not found: {args.lora}")

    # Built on demand rather than up front. Whether a spec needs img2img or text-to-image 
    # is a per-spec question. A spec with no _flat.png falls
    # back to ControlNet-only so deciding once from no-flat leaves the
    # other pipeline as None and the fallback crashes. 
    _pipes = {}

    def get_pipe(kind):
        if kind not in _pipes:
            cls = (StableDiffusionXLControlNetImg2ImgPipeline if kind == "i2i"
                   else StableDiffusionXLControlNetPipeline)
            p = cls.from_pretrained(BASE, **common).to("cuda")
            p.set_progress_bar_config(leave=False)
            if args.lora:
                print(f"  LoRA: {args.lora}")
                merge_lora(p, args.lora, args.lora_multiplier)
            _pipes[kind] = p
        return _pipes[kind]

    for f in specs:
        flat = None if args.no_flat else f.with_name(f.stem + "_flat.png")
        ctrl, init, w, h = prep(f, flat, args.resolution)
        prompt = args.prompt or sidecar(f, ".txt", FALLBACK_PROMPT)
        negative = args.negative or sidecar(f, ".neg.txt", FALLBACK_NEG)
        out_dir = Path(args.out) if args.out else f.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        ctrl.save(out_dir / f"{f.stem}_control.png")

        mode = "controlnet+img2img" if init is not None else "controlnet only"
        if args.lora:
            mode += f" +lora@{args.lora_multiplier}"
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
                im = get_pipe("i2i")(image=init, control_image=ctrl,
                                     strength=args.strength, **kw).images[0]
            else:
                im = get_pipe("t2i")(image=ctrl, width=w, height=h,
                                     **kw).images[0]
            p = out_dir / f"{f.stem}{args.suffix}_{args.seed + i}.png"
            im.save(p)
            print(f"    {p}")

    print("\ndone")


if __name__ == "__main__":
    main()
