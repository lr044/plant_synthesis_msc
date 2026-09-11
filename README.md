# Plant image synthesis, MSc project code

T.P. England, Durham University. Supervisor P. Remagnino.

Code for the LoRA fine-tuning experiments, the structure-first pipeline and
the automated evaluation. Every path goes through `paths.py`; There are no hard coded drive locations. Notebook outputs are the original run outputs,
left and have remained unchanged.

## Setup

```bash
export PLANT_ROOT=$(pwd)
pip install -r requirements.txt
python tools/check_refs.py       # every path resolves
python tools/expand_paths.py     # fill in ${ROOT} for OneTrainer
```

Images and LoRA weights are not in the repo. Restore them into `data/` and
`outputs/lora/` from the Zenodo archive; `data/README.md` has the layout.

## Layout

```
prep/        dataset construction
training/    per-species LoRA training, configs and concepts
structure/   geometric constructor and the ControlNet pipeline
eval/        automated metrics
data/        inputs, distributed separately
outputs/     weights, generated images, metrics, figures
tools/       path expansion and a reference checker
docs/        config notes and known gaps
```

## prep

`crop_bboxes.ipynb` crops individual plants out of the NI H2020 dunes subset
using the supplied boxes. `fetch_inat.ipynb` pulls research-grade iNaturalist
observations. `pool_species.ipynb` collapses the condition folders into one
folder per species for rating. `build_refset.ipynb` builds the 200-image
held-out set per species.

`docs/gaps.md`.

## training

Four backbones, one LoRA per species, twelve runs, all through
[OneTrainer](https://github.com/Nerogar/OneTrainer). The notebooks are thin
launchers; the configs carry the hyperparameters.

```
configs/<model>/<model>_colab_<species>.json   the run behind each LoRA
configs/superseded/                            earlier generic configs
configs/missing.md                             configs a notebook calls that were never exported
concepts/<model>/                              dataset and sample definitions
notebooks/                                     one launcher per backbone
exploratory/                                   Chroma and HiDream, not reported
```

Hyperparameters as they actually are, read off the configs:

| | SDXL | PixArt-Sigma | Qwen-Image | FLUX.2 |
|---|---|---|---|---|
| Rank / alpha | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 |
| LR | 3e-4 | 1e-4 | 1e-4 | 3e-4 |
| Resolution | 1024 | 1024 | 512 | 1024 |
| Batch / accum | 4 / 1 | 4 / 1 | 1 / 4 | 2 / 1 |
| Layer offload | 0 | 0 | 0.7 | 0.7 |
| Dtype | fp16 | bf16 | bf16 | bf16 |
| Epochs | 100 | 100 | 100 | 100 |


## structure

`build_plant.py` is the 2.5D constructor, `build_species.py` the
species-specific geometry. `gen_flat.py` runs ControlNet canny plus img2img
process over the flat colour map; `gen_flat_lora.py` does the same with a LoRA merged
in. `pipeline.ipynb` drives it end to end, `prompts.ipynb` 
is the prompt ablation behind Appendix A, `struct_lora.ipynb` the reported conditions.

`exploratory/` holds the real-background compositing route, built and not expanded upon
before the reported experiments due to poor integration with background.

## eval

`fid_kid.ipynb` provides FID and KID against the held-out and training sets plus
the real-vs-real floor (report Table 7). `per_image.ipynb` provides BioCLIP
zero-shot id, alignment, LPIPS diversity and nearest-neighbour distance
(report Table 6). `trait_probe.ipynb` is the linear probe on frozen BioCLIP
embeddings.

