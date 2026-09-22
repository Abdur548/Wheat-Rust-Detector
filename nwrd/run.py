"""Runs the full experiment: train -> evaluate -> benchmark -> figures -> compare.

    python -m nwrd.run --data-root /path/to/wheat_rust_patches --out runs
    python -m nwrd.run --stage compare --out runs        # re-aggregate only

Finished stages are skipped, and an interrupted training run resumes from last.pt, so the
same command can be repeated across Kaggle sessions until everything is done.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import fields

from .config import Config

STAGES = ('train', 'eval', 'bench', 'figures', 'compare')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', default='all', choices=('all',) + STAGES)
    p.add_argument('--out', dest='out_dir', default=Config.out_dir)
    p.add_argument('--bench-models', nargs='+', default=None,
                   help='default: the trained models plus canet_b4_twopass')
    p.add_argument('--bench-quick', action='store_true')
    for f in fields(Config):
        if f.name == 'out_dir':
            continue
        flag = '--' + f.name.replace('_', '-')
        default = f.default_factory() if callable(f.default_factory) else f.default
        if isinstance(default, bool):
            p.add_argument(flag, dest=f.name, type=lambda s: s.lower() in ('1', 'true', 'yes'),
                           default=default)
        elif isinstance(default, list):
            p.add_argument(flag, dest=f.name, nargs='+', type=type(default[0]), default=default)
        else:
            p.add_argument(flag, dest=f.name, type=type(default), default=default)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.out_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        log_file.write(line + '\n'); log_file.flush()

    stages = STAGES if a.stage == 'all' else (a.stage,)
    needs_data = {'train', 'eval', 'figures'} & set(stages)
    if needs_data:
        from .data import find_data_root
        cfg.data_root = find_data_root(cfg.data_root or None)
        log(f'data: {cfg.data_root}')
    with open(os.path.join(cfg.out_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Seed-major order: after each seed finishes, every model has a paired run to compare.
    runs = [(m, s, os.path.join(cfg.out_dir, m, f'seed{s}')) for s in cfg.seeds for m in cfg.models]
    complete = True
    for model, seed, run_dir in runs:
        if 'train' in stages and not os.path.exists(os.path.join(run_dir, 'TRAINED')):
            from .train import train_run
            if not train_run(cfg, model, seed, run_dir, log=log):
                complete = False
                break          # out of time budget: stop here, resume next session
        if 'eval' in stages and os.path.exists(os.path.join(run_dir, 'TRAINED')) \
                and not os.path.exists(os.path.join(run_dir, 'metrics.json')):
            from .evaluate import evaluate_run
            evaluate_run(cfg, model, run_dir, log=log)

    if 'bench' in stages and not os.path.exists(os.path.join(cfg.out_dir, 'benchmark.json')):
        from .benchmark import run_benchmarks
        models = a.bench_models or list(dict.fromkeys(cfg.models + ['canet_b4_twopass']))
        run_benchmarks(models, os.path.join(cfg.out_dir, 'benchmark.json'), size=cfg.img_size,
                       quick=a.bench_quick, log=log)

    if 'figures' in stages:
        from .evaluate import plot_qualitative
        first = {m: os.path.join(cfg.out_dir, m, f'seed{cfg.seeds[0]}') for m in cfg.models}
        first = {m: d for m, d in first.items() if os.path.exists(os.path.join(d, 'metrics.json'))}
        if first:
            os.makedirs(os.path.join(cfg.out_dir, 'figures'), exist_ok=True)
            plot_qualitative(cfg, first, os.path.join(cfg.out_dir, 'figures', 'qualitative_results.png'))

    if 'compare' in stages:
        from .compare import compare
        compare(cfg.out_dir, log=log)
    if not complete:
        log('INCOMPLETE: re-run the same command in a new session to continue.')
    return 0 if complete else 3


if __name__ == '__main__':
    sys.exit(main())
