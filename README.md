# Wheat Rust Disease Detection

A deep-learning system that segments wheat rust lesions pixel-by-pixel from leaf
photographs, turning a visual severity assessment normally made by eye into a
reproducible measurement of infected leaf area.

## Results

CANet with an EfficientNet-B4 encoder, evaluated on the validation split of the
`abdur548/nwrd-patched` dataset (Kaggle). Metrics are pooled over every pixel in the
split at a binarisation threshold of 0.79, selected by an F1 sweep over 0.25–0.79 on
that same split.

| Metric | Value |
| --- | --- |
| IoU | 0.7823 |
| Dice / F1 | 0.8779 |
| Precision | 0.8649 |
| Recall | 0.8913 |
| Specificity | 0.9892 |
| Accuracy | 0.9822 |

| Split | 512×512 patches | Notes |
| --- | --- | --- |
| Train | 3,153 of 8,160 available | 2,597 rust-positive, plus 556 background patches (10% of 5,563) |
| Validation | 480 | 124 contain rust; 7.2% of all pixels are rust |
| Test | TODO | present in the dataset, not evaluated by any code in this repository |

Three caveats a reader should weigh: the threshold was selected on the same split the
metrics are reported on; it sits at the top of the swept range, so the F1 optimum may
lie beyond 0.79; and no held-out test evaluation has been run. Because rust covers only
7.2% of validation pixels, accuracy and specificity are inflated by the class imbalance
— IoU and Dice are the meaningful figures.

Source: `backend/results/results.json`, written by
`backend/models/DL_Project_sprint2.ipynb` cells 17-18. Every figure above is
reproducible from the pixel counts in `backend/viusalisations/confusion_matrix (2).png`.

## Method

- **Architecture** — ImageNet-pretrained EfficientNet-B4 encoder → 1×1 reduction (1792→512) → chained context aggregation module (serial global flow with dilations 2/4/8, three parallel context flows at scales 2/4/8, attention-guided re-fusion) → asymmetric decoder fusing 1/32 context with 1/4 low-level features → single-channel head upsampled ×4. 29,524,784 parameters, 512×512 input.
- **Loss** — 0.7 × Dice + 0.3 × BCE-with-logits, with `pos_weight` 2.96 computed from the training masks.
- **Optimisation** — AdamW (lr 5e-5, weight decay 1e-4), CosineAnnealingWarmRestarts (T_0=5, T_mult=2, eta_min=1e-7), mixed precision, gradient clipping at 1.0.
- **Training** — 15 epochs, batch size 8, encoder frozen for the first 3 epochs; the checkpoint with the best validation IoU is kept.
- **Augmentation** — resize to 512, horizontal and vertical flips, 90° rotations, brightness/contrast jitter, elastic transform, ImageNet normalisation.

## Reproduce

TODO — there is no single-command evaluation script in this repository yet.

The table above was produced by running `backend/models/DL_Project_sprint2.ipynb` end to
end on a Colab GPU runtime. The notebook downloads the dataset itself through
`kagglehub`; cell 17 sweeps the threshold across the validation split, and cell 18
computes the final metrics and writes `results.json`.

## Setup

### 1. Download Model Weights
Before starting the backend, you need to download the required PyTorch `.pth` model files and place them inside the `backend/` directory:
- `best_model.pth`: The trained custom model weights.
Link:https://drive.google.com/file/d/1KvrcrUshGom8CA9JmjQkAXsIknWTwlRp/view?usp=sharing

- `efficientnet-b4-6ed6700e.pth`: The pre-trained EfficientNet-B4 backbone. Download it from [here](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth).

Ensure both files are present in the `backend/` folder.

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages using the `requirements.txt` file:

```powershell
pip install -r requirements.txt
```

### 3. Start the Backend
Open a terminal in the root directory. Activate the python virtual environment (if using one), and start the FastAPI server:

```powershell
# Activate virtual environment (optional)
.\venv\Scripts\Activate

# Start the backend server
uvicorn backend.main:app --reload
```
The API will run at `http://localhost:8000`.

### 3. Start the Frontend
Open a new terminal in the `frontend` directory. Install the npm dependencies and start the Vite dev server:

```powershell
cd frontend

# Run the development server
npm run dev
```
The frontend will run at `http://localhost:5173`.
