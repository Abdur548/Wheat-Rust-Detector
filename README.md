# Wheat Rust Disease Detection

A deep-learning system that segments wheat rust lesions pixel-by-pixel from leaf
photographs, turning a visual severity assessment normally made by eye into a
reproducible measurement of infected leaf area.

**Proposed method:** CANet with an ImageNet-pretrained EfficientNet-B4 encoder.
**Baseline:** the NWRD dataset paper — Anwar et al., *The NWRD Dataset*, Sensors 23(15):6942,
2023 (`sensors-23-06942.pdf`) — a plain UNet with adaptive patching with feedback (APF).
Its architecture is re-trained here under the same protocol as CANet (`unet`). DeepLabV3+
(ResNet-50) from the internship report is kept as a second reference, and DeepLabV3+
(EfficientNet-B4) is an optional same-encoder ablation.

Detailed write-ups live in [`md/`](md/): [`md/compare.md`](md/compare.md) (results,
architecture, pipeline differences and findings, with reasoning) and
[`md/paper_summary.md`](md/paper_summary.md) (the NWRD paper summarised, with its figures), and
[`md/improvement_summary.md`](md/improvement_summary.md) (what CANet-B4 changes and why, with
each claim marked measured, reasoned or pending).

## Results

**Pending the re-run.** Final numbers come from `notebooks/nwrd_experiments.ipynb` and land
in `results/comparison.md`.

The figures published so far cannot be used to compare the two models:

| Source | Model | IoU | Dice | Why it is not comparable |
| --- | --- | --- | --- | --- |
| Internship report | DeepLabV3+ R50 | 0.8304 | 0.9074 | different preprocessing; threshold tuned on the reported split |
| Sprint-2 notebook output | CANet-B4 | 0.8116 | 0.8960 | threshold tuned on the reported split; 15-epoch run |
| `backend/results/results.json` | CANet-B4 | 0.7823 | 0.8779 | a different run from the notebook output; threshold tuned on the reported split |

The paper's published UNet + APF results, for reference: **P 0.506, R 0.624, F1 0.557** on
its full test set (Table 3), and **P 0.593, R 0.552, F1 0.564, IoU 0.438** with background-only
patches removed (Table 4). These are measured on 128px patches of 10 held-out field images,
a different test unit from ours, so they are not comparable directly with our numbers. That
is why the paper's UNet is re-trained under our protocol. The pipeline also reports a
Table-4-style metric (rust-positive test patches only).

All three evaluate on the validation split, using a threshold picked on that same split.
None touches the test split, and the two models were trained with different
background-sampling ratios.

## Evaluation protocol

Implemented in `nwrd/`; every model gets identical treatment.

- **Data** — `abdur548/nwrd-patched` (Kaggle), 512×512 patches. Training uses every
  rust-positive patch plus 20% of background-only patches, sampled with a fixed seed so all
  models and seeds see the same set. Val and test are used in full.
- **Training** — 30 epochs, batch 8, AdamW (lr 5e-5, wd 1e-4), CosineAnnealingWarmRestarts
  (T_0 5, T_mult 2), fp16 mixed precision, gradient clip 1.0, pretrained encoders frozen for
  the first 3 epochs (the from-scratch UNet is never frozen). Loss 0.7·Dice + 0.3·BCE with `pos_weight` from the training masks.
- **Model selection** — the epoch with the best validation IoU at threshold 0.5.
- **Threshold** — chosen on **val** (IoU-maximising, grid 0.05–0.95), then frozen.
- **Reported metrics** — **test** split, pooled over every pixel, at the val threshold and at
  a fixed 0.5. Mean ± std over 3 seeds.
- **Significance** — paired image-level bootstrap (5,000 resamples) of the test IoU/Dice
  difference, plus a seed-level paired t-test.
- **Efficiency** — parameters, FLOPs, GPU latency (bs 1, fp32/fp16), throughput and peak
  memory (bs 8, fp16), CPU latency, and training time per epoch, all measured on the same
  device.

Fixes relative to the sprint notebooks:
- The per-epoch IoU averaged per batch, scoring a correctly predicted background-only batch
  as 0. That is why the training curves sat near 0.58 while pooled IoU was about 0.81.
  Metrics are now pooled.
- CANet ran its encoder twice per forward pass (`extract_features` plus
  `extract_endpoints`). It now runs once. Outputs are bit-identical and the encoder cost is
  halved; the benchmark reports both versions.

## Reproduce

**Kaggle (recommended).** Upload `notebooks/nwrd_experiments.ipynb`, attach the
`abdur548/nwrd-patched` dataset, pick GPU T4 and Internet on, then *Save & Run All*. The
notebook embeds the pipeline, so nothing needs cloning. Runs resume across sessions; the
instructions are in its first cell. Budget roughly 18–30 GPU-hours for 4 runs per seed × 3 seeds (set `TUNE_UNET_LR = False` in
the notebook to drop the tuned-lr UNet run and save a quarter).

**Locally, with a GPU:**

```bash
pip install -r requirements.txt
python -m nwrd.run --data-root /path/to/wheat_rust_patches --out runs
python -m nwrd.run --out runs --stage compare      # re-aggregate only
python -m pytest tests -q                          # CPU smoke tests, ~2 min
```

After editing anything in `nwrd/`, run `python scripts/build_notebook.py` to regenerate the
notebook.

**Taking results back into the app:** from the run output, copy `export/results.json` to
`backend/results/`, `export/best_model.pth` to `backend/models/`, and `export/*.png` to
`backend/visualisations/`. Copy `comparison.md`, `comparison.json` and `benchmark.json` to
`results/`. The exported model is the CANet seed with the best *validation* IoU.

## Layout

```
nwrd/            models, data, training, evaluation, benchmark, comparison (shared by everything)
notebooks/       nwrd_experiments.ipynb (generated); archive/ holds the sprint notebooks
scripts/         build_notebook.py; extract_paper_figures.py (figures for md/paper_summary.md)
md/              compare.md, paper_summary.md, improvement_summary.md, figures/paper/ (images extracted from the paper)
tests/           CPU smoke tests on synthetic data
backend/         FastAPI inference API; models/best_model.pth (Git LFS)
frontend/        React + Vite UI
dashboard.py     Streamlit alternative to the React UI
```

## Running the app

1. **Weights** — `backend/models/best_model.pth` is stored with Git LFS (`git lfs pull`), or
   download it from
   [Google Drive](https://drive.google.com/file/d/1KvrcrUshGom8CA9JmjQkAXsIknWTwlRp/view?usp=sharing).
   It contains the full network, so no separate EfficientNet download is needed.
2. **Dependencies** — Python 3.9+: `pip install -r requirements.txt`.
3. **Backend** — `uvicorn backend.main:app --reload`, serving at `http://localhost:8000`.
   The prediction threshold defaults to the val-selected value in
   `backend/results/results.json`.
4. **Frontend** — `cd frontend && npm install && npm run dev`, at `http://localhost:5173`.
5. **Or the Streamlit dashboard** — `streamlit run dashboard.py`.
