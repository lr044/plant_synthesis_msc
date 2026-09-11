"""Every path in the project. No absolute Drive paths anywhere else.

ROOT is this file's directory unless PLANT_ROOT is set, so the tree works
from a clone, a Colab mount or a local checkout without editing anything.

OneTrainer configs hold a ${ROOT} token because OneTrainer reads the
JSON straight off disk and does no substitution. Run tools/expand_paths.py
before training.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PLANT_ROOT") or Path(__file__).resolve().parent)

SPECIES = ["achillea", "carpobrotus", "eryngium"]
NAMES = {
    "achillea": ("Achillea", "maritima"),
    "carpobrotus": ("Carpobrotus", "acinaciformis"),
    "eryngium": ("Eryngium", "maritimum"),
}

DATA = ROOT / "data"
RAW = DATA / "raw"
NIH = RAW / "nih2020"
INAT = RAW / "inaturalist"
TRAIN = DATA / "train"
REF = DATA / "reference"
BG = DATA / "backgrounds"
NOTES = DATA / "annotations"

OUT = ROOT / "outputs"
LORA = OUT / "lora"
GEN = OUT / "generated"
BASE = OUT / "generated_baseline"
STRUCT = OUT / "structure_first"
METRICS = OUT / "metrics"
FIGS = OUT / "figures"
RATINGS = OUT / "ratings"
WORK = OUT / "workspace"
CACHE = OUT / "workspace-cache"

CONFIGS = ROOT / "training/configs"
CONCEPTS = ROOT / "training/concepts"

_STEM = {"sdxl": "sdxl_colab", "pixart": "pixartsigma_colab",
         "qwen": "qwen_colab", "flux2": "flux2_colab"}


def lora(model, sp):
    return LORA / f"{model}_{sp}" / "lora.safetensors"


def config(model, sp):
    return CONFIGS / model / f"{_STEM[model]}_{sp}.json"


def concepts(model, sp):
    return CONCEPTS / model / f"train_concepts_{sp}.json"


for d in (DATA, RAW, TRAIN, REF, OUT, LORA, GEN, BASE, STRUCT, METRICS, FIGS, RATINGS):
    d.mkdir(parents=True, exist_ok=True)
