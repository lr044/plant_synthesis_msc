# Outputs

Written by the code here. Weights and images go via Zenodo, not git.

```
lora/<model>_<species>/lora.safetensors   12 adapters
generated/<model>_<species>/              100 images per LoRA condition
generated_baseline/<model>_<species>/     100 per unadapted condition
structure_first/                          structure-first outputs
metrics/                                  CSVs from eval/
figures/                                  report figures
ratings/                                  botanist rating export
workspace/, workspace-cache/              OneTrainer scratch, deletable
```

Folder names under `generated/` and `generated_baseline/` have to contain the
species key. The metric notebooks accept the model name 
