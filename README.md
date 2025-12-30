# Multispectral Coastal Habitat Segmentation (Project 3)

This repo is a cleaned, GitHub-ready version of my **Project 3** work: semantic segmentation of coastal habitat from **8-band GeoTIFF tiles** into **7 classes**.

What was in the original submission zip:
- a PDF report
- a Colab notebook *snippet* (not fully runnable because pieces were omitted for the submission)

What this repo adds:
- a **reproducible training + inference pipeline** (`train.py`, `infer.py`)
- a `src/` package with dataset + model code
- clear instructions + a sane `.gitignore`

## Classes
The project assumes 7 habitat classes. If your dataset uses different label IDs or RGB colors, edit `CLASS_COLORS` in `src/dataset.py`.

## Quickstart

### 1) Create an environment (recommended: conda)
```bash
conda env create -f environment.yml
conda activate habitat-seg
```

### 2) Put your data in this layout
```
data/
  raw_labeled_data/
    images/        # GeoTIFF image tiles (.tif), 8 channels
    annotations/   # GeoTIFF masks (.tif) either label-IDs or RGB-coded
```

### 3) Train
```bash
python train.py --data_dir data/raw_labeled_data --epochs 50 --batch_size 16
```

### 4) Run inference on a folder
```bash
python infer.py --checkpoint checkpoints/best.pt --image_dir data/raw_labeled_data/images --out_dir outputs
```

## Notes (read this if you want recruiters to take it seriously)
- **Do not commit the dataset**. Keep it local and document how to obtain it.
- Add 2–3 **sample predictions** (input / ground truth / prediction) into `assets/` and embed them in this README.
- If you trained in Colab, mention GPU + runtime and include your best metrics.

## Files
- `report/PROJECT-3_REPORT.pdf` — original report
- `notebooks/original_submission.ipynb` — original notebook artifact
- `src/` — dataset + model code
- `train.py` — training entrypoint
- `infer.py` — inference entrypoint

