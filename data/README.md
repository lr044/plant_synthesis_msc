# Data

Distributed through Zenodo, not git. Restore into this tree; `paths.py`
resolves all of it.

```
raw/nih2020/          NI H2020 dunes subset as supplied, YOLO labels
raw/inaturalist/      research-grade observations, one folder per species

reference/<species>/  200 held-out real images each
backgrounds/tiles/    1,400 habitat tiles, exploratory route only
annotations/          species-level trait annotations
```

Reference must never overlap training. The leakage check in `eval/fid_kid.ipynb`
exists for this purpose. Run it before distributional metrics.
