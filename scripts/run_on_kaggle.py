"""Runs notebooks/nwrd_experiments.ipynb on Kaggle GPU from the command line.

Needs a Kaggle API token once: kaggle.com -> Settings -> API -> Create New Token, then save
the downloaded kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json (or ~/.kaggle/kaggle.json).

    pip install kaggle
    python scripts/run_on_kaggle.py --watch            # push, then poll until it finishes
    python scripts/run_on_kaggle.py --resume --watch   # continue from the last version's runs/
    python scripts/run_on_kaggle.py --status           # check without pushing

The kernel is private, GPU-enabled, with internet on and abdur548/nwrd-patched attached.
`--resume` attaches the previous version's output, which the notebook's restore cell copies
into place, so training continues instead of restarting. Repeat until the log has no
INCOMPLETE line. Output lands in runs_from_kaggle/.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'nwrd_experiments.ipynb')
SLUG = 'nwrd-canet-b4-vs-baselines'  # must match the slug Kaggle derives from the title
PUSH_DIR = os.path.join(ROOT, '.kaggle_push')
OUT_DIR = os.path.join(ROOT, 'runs_from_kaggle')
DONE_STATES = ('complete', 'error', 'cancelAcknowledged', 'cancelRequested')


def kaggle(*args, check=True):
    """Runs the kaggle CLI through the current interpreter, so no PATH setup is needed."""
    cmd = [sys.executable, '-m', 'kaggle', *args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or '') + (p.stderr or '')
    if check and p.returncode != 0:
        raise SystemExit(f'kaggle {" ".join(args)} failed:\n{out.strip()}')
    return out.strip()


def username():
    for path in (os.path.expanduser('~/.kaggle/kaggle.json'),
                 os.path.join(os.environ.get('KAGGLE_CONFIG_DIR', ''), 'kaggle.json')):
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)['username']
    if os.environ.get('KAGGLE_USERNAME'):
        return os.environ['KAGGLE_USERNAME']
    raise SystemExit('No Kaggle credentials. Save kaggle.json to ~/.kaggle/ '
                     '(kaggle.com -> Settings -> API -> Create New Token).')


def push(kernel_id, resume):
    os.makedirs(PUSH_DIR, exist_ok=True)
    meta = {
        'id': kernel_id,
        'title': 'NWRD CANet-B4 vs baselines',
        'code_file': 'nwrd_experiments.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
        'dataset_sources': ['abdur548/nwrd-patched'],
        'competition_sources': [],
        # Attaching the kernel's own previous output lets the notebook resume.
        'kernel_sources': [kernel_id] if resume else [],
    }
    with open(os.path.join(PUSH_DIR, 'kernel-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(NOTEBOOK, encoding='utf-8') as src, \
            open(os.path.join(PUSH_DIR, 'nwrd_experiments.ipynb'), 'w', encoding='utf-8') as dst:
        dst.write(src.read())
    print(kaggle('kernels', 'push', '-p', PUSH_DIR))


def status(kernel_id):
    text = kaggle('kernels', 'status', kernel_id, check=False)
    for state in ('complete', 'error', 'running', 'queued', 'cancel'):
        if state in text.lower():
            return state, text
    return 'unknown', text


def fetch_output(kernel_id):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(kaggle('kernels', 'output', kernel_id, '-p', OUT_DIR, check=False))
    comparison = os.path.join(OUT_DIR, 'runs', 'comparison.md')
    if os.path.exists(comparison):
        print('\n' + open(comparison, encoding='utf-8').read())
    else:
        print(f'no comparison.md yet in {OUT_DIR}')
    log = os.path.join(OUT_DIR, 'runs', 'log.txt')
    if os.path.exists(log):
        tail = open(log, encoding='utf-8').read().splitlines()[-15:]
        print('--- log tail ---\n' + '\n'.join(tail))
        if any('INCOMPLETE' in line for line in tail):
            print('\nStopped at the time budget: re-run with --resume --watch to continue.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resume', action='store_true', help="attach the kernel's previous output")
    ap.add_argument('--watch', action='store_true', help='poll until the run finishes')
    ap.add_argument('--status', action='store_true', help='report status only, do not push')
    ap.add_argument('--poll-minutes', type=float, default=10)
    a = ap.parse_args()

    kernel_id = f'{username()}/{SLUG}'
    if a.status:
        state, text = status(kernel_id)
        print(text)
        if state == 'complete':
            fetch_output(kernel_id)
        return

    push(kernel_id, a.resume)
    print(f'https://www.kaggle.com/code/{kernel_id}')
    if not a.watch:
        print('Check later:  python scripts/run_on_kaggle.py --status')
        return

    while True:
        time.sleep(a.poll_minutes * 60)
        state, text = status(kernel_id)
        print(f'{time.strftime("%H:%M")} {state}')
        if state in ('complete', 'error') or any(d in text for d in DONE_STATES):
            print(text)
            fetch_output(kernel_id)
            return


if __name__ == '__main__':
    main()
