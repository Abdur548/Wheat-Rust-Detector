"""Builds notebooks/nwrd_experiments.ipynb: a self-contained Kaggle/Colab notebook that
embeds the nwrd/ package, so it runs without cloning the repository.

    python scripts/build_notebook.py

Re-run after changing anything in nwrd/ so the notebook stays in sync.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ['__init__', 'config', 'models', 'data', 'metrics', 'train', 'evaluate',
           'benchmark', 'compare', 'run']

INTRO = """# NWRD wheat-rust segmentation: CANet-B4 vs baselines

Trains every model under one protocol, picks the threshold on **val**, reports on the held-out
**test** split, benchmarks efficiency, and writes the comparison to `runs/comparison.md`.

**Kaggle setup**
1. *Add Input* → dataset **`abdur548/nwrd-patched`**.
2. *Settings* → Accelerator **GPU T4 x2** (one GPU is used; P100 may not be supported by
   Kaggle's current PyTorch build), **Internet on** (pip and ImageNet weights).
3. *Save Version* → **Save & Run All (Commit)**. It runs in the background for up to 12 h.

**If it stops at the time budget** (the log ends in `INCOMPLETE`): open the notebook, *Add Input* →
*Your Work* → this notebook's latest version, and commit again. The restore cell copies the
previous `runs/` and every run resumes where it stopped. Repeat until the log has no `INCOMPLETE`.

**Colab**: Runtime → GPU. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab secrets (🔑 sidebar)
so the dataset can be downloaded. Never paste keys into the notebook itself.

Rough cost on a T4: 1.5–2.5 h per run × 4 runs per seed (3 with `TUNE_UNET_LR = False`) × 3 seeds,
so plan on two or three sessions (Kaggle's weekly quota is 30 GPU-h).
Runs go seed by seed, so a paired comparison exists as soon as the first seed finishes."""

CONFIG = """# ── Experiment settings ─────────────────────────────────────────────────────
MODELS = ['canet_b4', 'unet', 'deeplabv3plus_r50']   # unet = NWRD paper baseline; add 'deeplabv3plus_b4' for the same-encoder ablation
# The shared lr (5e-5) suits fine-tuning pretrained encoders and may under-train the
# from-scratch UNet, which would flatter CANet. True adds a UNet run with its own lr; the
# report then uses whichever UNet run is better on validation. Costs ~2-3 GPU-h per seed.
TUNE_UNET_LR = True
UNET_TUNED_LR = 1e-3
if TUNE_UNET_LR and 'unet_tuned' not in MODELS:
    MODELS.append('unet_tuned')
SEEDS = [42, 43, 44]
EPOCHS = 30
TIME_BUDGET_HOURS = 11.0    # Kaggle kills sessions at 12 h; stop cleanly before that
NUM_WORKERS = 4
OUT = '/kaggle/working/runs' if os.path.exists('/kaggle') else '/content/runs'"""

SETUP = """import os, sys, glob, shutil, subprocess
!pip install -q segmentation-models-pytorch==0.5.0 efficientnet_pytorch==0.7.1 albumentations
import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'enable a GPU accelerator'
os.makedirs('code/nwrd', exist_ok=True)
sys.path.insert(0, os.path.abspath('code'))"""

DATA = """# Kaggle mounts the dataset under /kaggle/input; Colab downloads it with kagglehub.
DATA_ROOT = ''
if not os.path.exists('/kaggle/input'):
    from google.colab import userdata
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    import kagglehub
    DATA_ROOT = kagglehub.dataset_download('abdur548/nwrd-patched')
from nwrd.data import find_data_root
DATA_ROOT = find_data_root(DATA_ROOT or None)
print('dataset:', DATA_ROOT, '|', {s: len(os.listdir(os.path.join(DATA_ROOT, s, 'images'))) for s in ('train', 'val', 'test')})
# How were the splits made? If patches of one field image appear in more than one split, the
# test scores are inflated by leakage. Inspect the dataset's own notes and file naming.
for f in ('README.md', 'processing_stats.json'):
    p = os.path.join(DATA_ROOT, f)
    if os.path.exists(p):
        print(f'--- {f} ---'); print(open(p, encoding='utf-8', errors='replace').read()[:3000])
for s in ('train', 'val', 'test'):
    print(s, sorted(os.listdir(os.path.join(DATA_ROOT, s, 'images')))[:5])"""

RESTORE = """# Continue from a previous session: copy runs/ from any attached earlier notebook version.
prev = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/runs/config.json', recursive=True)]
if prev and not os.path.exists(OUT):
    shutil.copytree(prev[0], OUT)
    print('restored previous runs from', prev[0])
    for d in sorted(glob.glob(f'{OUT}/*/seed*')):
        state = 'done' if os.path.exists(f'{d}/metrics.json') else 'trained' if os.path.exists(f'{d}/TRAINED') \\
            else 'partial' if os.path.exists(f'{d}/last.pt') else 'empty'
        print(f'  {d[len(OUT) + 1:]}: {state}')
else:
    print('starting fresh' if not prev else f'{OUT} already exists; not restoring')"""

RUN = """cmd = [sys.executable, '-m', 'nwrd.run', '--data-root', DATA_ROOT, '--out', OUT,
       '--models', *MODELS, '--seeds', *map(str, SEEDS), '--epochs', str(EPOCHS),
       '--time-budget-hours', str(TIME_BUDGET_HOURS), '--num-workers', str(NUM_WORKERS),
       '--unet-tuned-lr', str(UNET_TUNED_LR)]
print(' '.join(cmd))
proc = subprocess.Popen(cmd, cwd='code', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end='')
print('exit code', proc.wait(), '(3 = stopped at time budget; commit again to resume)')"""

SHOW = """from IPython.display import Markdown, Image, display
if os.path.exists(f'{OUT}/comparison.md'):
    display(Markdown(open(f'{OUT}/comparison.md', encoding='utf-8').read()))
    for f in ('training_curves.png', 'qualitative_results.png'):
        if os.path.exists(f'{OUT}/figures/{f}'):
            display(Image(f'{OUT}/figures/{f}'))"""

OUTRO = """## Taking the results back to the repo

Download the notebook output and copy:

| From `runs/` | To the repo |
|---|---|
| `comparison.md`, `comparison.json`, `benchmark.json` | `results/` |
| `export/results.json` | `backend/results/results.json` |
| `export/best_model.pth` | `backend/models/best_model.pth` |
| `export/*.png` | `backend/visualisations/` |

`export/` holds the CANet seed with the best **validation** IoU; test metrics are never used
to choose it."""


def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.splitlines(True)}


def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': src.splitlines(True)}


def main():
    cells = [md(INTRO), code(SETUP), code(CONFIG),
             md('### Pipeline source (generated from `nwrd/` by `scripts/build_notebook.py`; edit there)')]
    for m in MODULES:
        with open(os.path.join(ROOT, 'nwrd', f'{m}.py'), encoding='utf-8') as f:
            cells.append(code(f'%%writefile code/nwrd/{m}.py\n' + f.read()))
    cells += [md('### Data and resume'), code(DATA), code(RESTORE),
              md('### Train → evaluate → benchmark → compare'), code(RUN), code(SHOW), md(OUTRO)]
    for i, c in enumerate(cells):
        c['id'] = f'cell-{i:02d}'
    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                       'language_info': {'name': 'python'},
                       'accelerator': 'GPU', 'kaggle': {'accelerator': 'nvidiaTeslaT4',
                                                        'isInternetEnabled': True}}}
    out = os.path.join(ROOT, 'notebooks', 'nwrd_experiments.ipynb')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
