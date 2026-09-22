import json
import sys
from pathlib import Path
from typing import Optional
import base64
import io
import torch
import numpy as np
from PIL import Image as PILImage
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Wheat Rust Detection API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
# The architecture lives in nwrd/models.py, shared with training. The checkpoint holds the
# encoder weights too, so no separate EfficientNet download is needed.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nwrd.models import build_model, load_checkpoint  # noqa: E402

BACKEND = ROOT / 'backend'
CHECKPOINT = BACKEND / 'models' / 'best_model.pth'
RESULTS = BACKEND / 'results' / 'results.json'
VIS_DIR = BACKEND / 'visualisations'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None


def default_threshold():
    """The validation-selected threshold recorded with the metrics, else 0.5."""
    try:
        return float(json.loads(RESULTS.read_text())['best_threshold'])
    except (OSError, KeyError, ValueError):
        return 0.5


@app.on_event("startup")
def load_model():
    global model
    if not CHECKPOINT.exists():
        print(f"Error: no checkpoint at {CHECKPOINT}. /api/predict will return 500.")
        return
    model = build_model('canet_b4', pretrained=False).to(device)
    load_checkpoint(model, CHECKPOINT, map_location=device)
    model.eval()
    print(f"Model loaded from {CHECKPOINT} on {device}")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/results")
def get_results():
    if not RESULTS.exists():
        raise HTTPException(status_code=404, detail=f"{RESULTS.relative_to(ROOT)} not found")
    return json.loads(RESULTS.read_text())

@app.get("/api/images/{name}")
def get_image(name: str):
    filepath = VIS_DIR / Path(name).name
    if filepath.suffix == '.png' and filepath.exists():
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Image not found")

def image_to_base64(img_arr, mode='RGB'):
    img = PILImage.fromarray(img_arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), threshold: Optional[float] = Form(None)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if threshold is None:
        threshold = default_threshold()

    # Read image
    contents = await file.read()
    try:
        img_raw = PILImage.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Preprocess
    img_resized = img_raw.resize((512, 512), PILImage.BILINEAR)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    
    mean_t = [0.485, 0.456, 0.406]
    std_t  = [0.229, 0.224, 0.225]
    for c in range(3):
        img_arr[:,:,c] = (img_arr[:,:,c] - mean_t[c]) / std_t[c]
        
    inp = torch.tensor(img_arr).permute(2,0,1).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        pred_prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
        
    pred_bin = (pred_prob >= threshold).astype(np.uint8)
    
    # Generate Output Images
    display_img = np.array(img_resized)
    overlay = display_img.copy()
    rust_px = pred_bin == 1
    overlay[rust_px] = (overlay[rust_px] * 0.4 + np.array([255, 60, 60]) * 0.6).astype(np.uint8)
    
    prob_map_display = (pred_prob * 255).astype(np.uint8)
    bin_mask_display = (pred_bin * 255).astype(np.uint8)
    
    # Convert to Base64
    resized_original_b64 = f"data:image/png;base64,{image_to_base64(display_img)}"
    prob_map_b64 = f"data:image/png;base64,{image_to_base64(prob_map_display, mode='L')}"
    pred_mask_b64 = f"data:image/png;base64,{image_to_base64(bin_mask_display, mode='L')}"
    overlay_b64 = f"data:image/png;base64,{image_to_base64(overlay)}"
    
    # Calculate Metrics
    total_pixels = pred_bin.size
    disease_px = int(pred_bin.sum())
    coverage_pct = (disease_px / total_pixels) * 100
    
    # Determine Severity
    if coverage_pct == 0: severity = "healthy"
    elif coverage_pct < 5: severity = "early"
    elif coverage_pct < 20: severity = "moderate"
    else: severity = "severe"
    
    return {
        "rust_pixels": disease_px,
        "total_pixels": total_pixels,
        "coverage_pct": round(coverage_pct, 2),
        "severity": severity,
        "threshold": threshold,
        "pred_mask_b64": pred_mask_b64,
        "prob_map_b64": prob_map_b64,
        "overlay_b64": overlay_b64,
        "resized_original_b64": resized_original_b64
    }
