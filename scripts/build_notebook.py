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

**Colab**: Runtime → GPU. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab secrets (🔑 sidebar);
if secrets are unavailable the notebook asks for them, and the key stays hidden. Never paste keys
into the notebook itself. Runs are written to `MyDrive/nwrd_runs` so a disconnect does not lose
them: re-running the notebook continues where it stopped. Free Colab sessions are shorter than
Kaggle's, so expect several sessions, or set `USE_DRIVE_ON_COLAB = False` for a throwaway run.

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
NUM_WORKERS = 4   # TIME_BUDGET_HOURS is set per platform in the next cell
# Colab wipes local disk on disconnect, so runs go to Google Drive and survive to be resumed.
# Set False to keep them in /content (lost when the runtime ends).
USE_DRIVE_ON_COLAB = True"""

STORAGE = """# Where runs are written. On Kaggle: /kaggle/working (kept with each committed version).
# On Colab: Google Drive, so a disconnect does not lose finished runs.
ON_KAGGLE = os.path.exists('/kaggle/working')
if ON_KAGGLE:
    OUT, TIME_BUDGET_HOURS = '/kaggle/working/runs', 11.0     # Kaggle kills sessions at 12 h
else:
    OUT, TIME_BUDGET_HOURS = '/content/runs', 3.0             # free Colab sessions are shorter
    if USE_DRIVE_ON_COLAB:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            OUT = '/content/drive/MyDrive/nwrd_runs'
        except Exception as e:
            print('Drive not mounted (%s) - runs stay in %s and are LOST on disconnect.' % (e, OUT))
os.makedirs(OUT, exist_ok=True)
print('runs ->', OUT, '| time budget', TIME_BUDGET_HOURS, 'h')"""

SETUP = """import os, sys, glob, shutil, subprocess
!pip install -q segmentation-models-pytorch==0.5.0 efficientnet_pytorch==0.7.1 albumentations kagglehub
import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'enable a GPU accelerator'
os.makedirs('code/nwrd', exist_ok=True)
sys.path.insert(0, os.path.abspath('code'))"""

DATA = """from nwrd.data import find_data_root

# Kaggle: the attached dataset already sits under /kaggle/input. Colab: download it.
# Deciding by "does /kaggle/input exist" is unreliable - some Colab runtimes have an empty
# /kaggle/input - so try to find the data first and only download if that fails.
try:
    DATA_ROOT = find_data_root(None)
except FileNotFoundError:
    import kagglehub
    if not os.environ.get('KAGGLE_KEY'):
        try:
            from google.colab import userdata      # Colab: key icon in the left sidebar
            os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
        except Exception as e:
            import getpass
            print('Colab secrets unavailable (%s); enter Kaggle credentials.' % e)
            os.environ['KAGGLE_USERNAME'] = input('Kaggle username: ').strip()
            os.environ['KAGGLE_KEY'] = getpass.getpass('Kaggle API key (hidden): ').strip()
    DATA_ROOT = find_data_root(kagglehub.dataset_download('abdur548/nwrd-patched'))

print('dataset:', DATA_ROOT, '|', {s: len(os.listdir(os.path.join(DATA_ROOT, s, 'images'))) for s in ('train', 'val', 'test')})
# How were the splits made? If patches of one field image appear in more than one split, the
# test scores are inflated by leakage. Inspect the dataset's own notes and file naming.
for f in ('README.md', 'processing_stats.json'):
    p = os.path.join(DATA_ROOT, f)
    if os.path.exists(p):
        print('--- %s ---' % f); print(open(p, encoding='utf-8', errors='replace').read()[:3000])
for s in ('train', 'val', 'test'):
    print(s, sorted(os.listdir(os.path.join(DATA_ROOT, s, 'images')))[:5])"""

RESTORE = """# Continue an earlier session. On Drive/Kaggle-working, finished runs are already in OUT.
prev = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/runs/config.json', recursive=True)]
if prev and not os.path.exists(os.path.join(OUT, 'config.json')):
    shutil.copytree(prev[0], OUT, dirs_exist_ok=True)
    print('restored previous runs from', prev[0])
done = sorted(glob.glob(os.path.join(OUT, '*', 'seed*')))
for d in done:
    state = ('done' if os.path.exists(os.path.join(d, 'metrics.json')) else
             'trained' if os.path.exists(os.path.join(d, 'TRAINED')) else
             'partial' if os.path.exists(os.path.join(d, 'last.pt')) else 'empty')
    print(' ', os.path.relpath(d, OUT), state)
print('%d run directories found' % len(done))"""

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
    cells = [md(INTRO), code(SETUP), code(CONFIG), code(STORAGE),
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
