"""
Export the training / test input dataset for the latest formulation
===================================================================
The experiment script (full_spectrum_cos_noflux.py) generates its PDE data
on the fly with fixed random seeds and never writes it to disk. This script
re-imports the *same* generator functions from that module and materialises
the dataset, so the exported arrays are bit-identical to what the models
were trained on.

Reproduction protocol (must not be altered)
    training : make_data(s, N_TRAIN, seed = 61000 + s)   s = 0..4
    test     : make_data(s, N_TEST,  seed = 60000 + s)   s = 0..4
    N_TRAIN = 400, N_TEST = 100, five time scales indexed 0..4

Outputs (written to ./dataset/)
    train_scale{s}.csv      400 rows per scale
    test_scale{s}.csv       100 rows per scale
    all_data.npz            all arrays in one bundle
    dataset_manifest.json   shapes, seeds, checksums

Columns
    x        normalised depth in [0,1]            (-)
    t_rel    elapsed time since the anchor        (s)
    t_abs    absolute timestamp                   (s, unix)
    T        temperature                          (degC)
    H        relative humidity                    (-)
    N P K    macro nutrients                      (-, normalised)
    Mg Ca S  secondary nutrients                  (-, normalised)
    Fe Mn Zn Cu B  micro nutrients                (-, normalised)
    phi      porosity                             (-, dimensionless volume fraction)
    u        reference microbial density          (-)
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.environ.get('SRC_DIR', HERE)
sys.path.insert(0, SRC_DIR)

# Import the generator functions without running the training block:
# full_spectrum_cos_noflux.py trains at import time, so we execute only the
# definitions we need by reading them through runpy with a guard.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    'fsn', os.path.join(SRC_DIR, 'full_spectrum_cos_noflux.py'))
_fsn = importlib.util.module_from_spec(_spec)


def _load_definitions_only():
    """Execute the module's source up to the training section only."""
    path = os.path.join(SRC_DIR, 'full_spectrum_cos_noflux.py')
    src = open(path, encoding='utf-8').read()
    cut = src.index('test_data={s:make_data')
    ns = {'__name__': 'fsn_defs'}
    exec(compile(src[:cut], path, 'exec'), ns)
    return ns


ns = _load_definitions_only()
make_data = ns['make_data']
N_TRAIN = ns['N_TRAIN']
N_TEST = ns['N_TEST']
SCALE_NAMES = ns['SCALE_NAMES']

ELEMS = ['N', 'P', 'K', 'Mg', 'Ca', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B']
COLS = ['x', 't_rel', 't_abs', 'T', 'H'] + ELEMS + ['phi', 'u']

OUT = os.path.join(HERE, 'dataset')
os.makedirs(OUT, exist_ok=True)


def frame(scale, n, seed):
    xs, t_rel, t_abs, T, H, M, phi, u = make_data(scale, n, seed)
    data = np.column_stack([xs, t_rel, t_abs, T, H, M, phi, u])
    df = pd.DataFrame(data, columns=COLS)
    df.insert(0, 'scale', SCALE_NAMES[scale])
    return df


def sha(df):
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:16]


manifest = {'protocol': {'N_TRAIN': int(N_TRAIN), 'N_TEST': int(N_TEST),
                         'train_seed': '61000 + scale_index',
                         'test_seed': '60000 + scale_index',
                         'scales': SCALE_NAMES},
            'columns': COLS, 'files': {}}

npz = {}
for s in range(5):
    tr = frame(s, N_TRAIN, 61000 + s)
    te = frame(s, N_TEST, 60000 + s)
    for tag, df in [('train', tr), ('test', te)]:
        fn = f'{tag}_scale{s}_{SCALE_NAMES[s]}.csv'
        df.to_csv(os.path.join(OUT, fn), index=False)
        manifest['files'][fn] = {'rows': int(len(df)), 'cols': int(df.shape[1]),
                                 'sha256_16': sha(df)}
    npz[f'train_x_{s}'] = tr['x'].values
    npz[f'train_t_rel_{s}'] = tr['t_rel'].values
    npz[f'train_u_{s}'] = tr['u'].values
    npz[f'test_x_{s}'] = te['x'].values
    npz[f'test_t_rel_{s}'] = te['t_rel'].values
    npz[f'test_u_{s}'] = te['u'].values

np.savez_compressed(os.path.join(OUT, 'all_data.npz'), **npz)
with open(os.path.join(OUT, 'dataset_manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print('=' * 70)
print('Dataset exported to', OUT)
print('=' * 70)
tot = 0
for fn, info in manifest['files'].items():
    tot += info['rows']
    print('  %-34s %4d rows  sha %s' % (fn, info['rows'], info['sha256_16']))
print('  all_data.npz                       %d arrays' % len(npz))
print('\n  total rows: %d  (train %d + test %d)'
      % (tot, 5 * N_TRAIN, 5 * N_TEST))

# sanity: reference field should be non-negative
u_all = np.concatenate([npz[f'train_u_{s}'] for s in range(5)])
print('  reference u: min=%.4f max=%.4f  (non-negative: %s)'
      % (u_all.min(), u_all.max(), bool((u_all >= 0).all())))
