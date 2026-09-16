"""Загрузка и разбор данных QED-FCI (молекулы и атом гелия)."""
import glob
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work')
MOL = os.path.join(ROOT, 'qedfci_frozen_cs+z_with_opt')
ATOM = os.path.join(ROOT, 'qedfci_atom')

RE = {'H2': 0.7414, 'LiH': 1.5957}


def _read(path):
    cols, meta, rows = None, {}, []
    for line in open(path):
        if line.startswith('#'):
            if 'columns' in line and '=' in line:
                cols = [c.strip() for c in line.split('=', 1)[1].split(',')]
            elif '=' in line:
                k, v = line[1:].split('=', 1)
                meta[k.strip()] = v.strip()
            continue
        p = line.split()
        if len(p) < 5 or 'FAILED' in line:
            continue
        rows.append(p)
    if not rows:
        return None, meta
    d = {}
    for i, c in enumerate(cols):
        if c == 'status':
            d[c] = np.array([r[i] if i < len(r) else '' for r in rows])
        else:
            d[c] = np.array([float(r[i]) for r in rows])
    return d, meta


def mol_set(system, regime, perp=False):
    """system: 'h2'|'lih'; regime: 'el'|'vib'. -> {lambda: dict}

    Perpendicular-orientation runs live in a separate directory for H2
    (fci_h2_*_perp) but are interleaved with the axial runs, distinguished
    only by a '_pol+...' filename suffix, for LiH (fci_lih_*). Both layouts
    are searched here; the pol+ filter picks out the right files regardless
    of which directory they came from.
    """
    base_dir = f'fci_{system}_{regime}'
    out = {}
    for d in (base_dir, base_dir + '_perp'):
        for f in sorted(glob.glob(os.path.join(MOL, d, '*_data.txt'))):
            base = os.path.basename(f)
            if perp and 'pol+' not in base:
                continue
            if (not perp) and 'pol+' in base:
                continue
            dat, meta = _read(f)
            if dat is None:
                continue
            out[round(float(meta['lambda']), 4)] = dat
    return out


def atom_set():
    """-> {(state, omega): dict}; state in {'g','t','s'}"""
    files = {
        ('g', 0.0293): 'he_ground/He_ms0_root0_om0.0293_aug-cc-pvtz+3d_data.txt',
        ('t', 0.0293): 'he_triplet/He_ms2_root0_om0.0293_aug-cc-pvtz+3d_data.txt',
        ('s', 0.0293): 'he_singlet/He_ms0_root1_om0.0293_aug-cc-pvtz+3d_data.txt',
        ('g', 0.1000): 'he_ground_w01/He_ms0_root0_om0.1000_aug-cc-pvtz+3d_data.txt',
        ('t', 0.1000): 'he_triplet_w01/He_ms2_root0_om0.1000_aug-cc-pvtz+3d_data.txt',
        ('s', 0.1000): 'he_singlet_w01/He_ms0_root1_om0.1000_aug-cc-pvtz+3d_data.txt',
    }
    out = {}
    for k, rel in files.items():
        dat, meta = _read(os.path.join(ATOM, rel))
        if dat is not None:
            out[k] = dat
    return out


def at(d, x, key, xkey='R', deg=3):
    """Локальная полиномиальная интерполяция d[key] в точку x."""
    X = d[xkey]
    if x < X.min() - 1e-9 or x > X.max() + 1e-9:
        return np.nan
    i = int(np.argmin(np.abs(X - x)))
    lo = max(0, min(i - deg // 2 - 1, len(X) - deg - 1))
    sl = slice(lo, lo + deg + 1)
    return float(np.polyval(np.polyfit(X[sl], d[key][sl], deg), x))


def common_grid(a, b, key, xkey='R'):
    """Относительное изменение b[key] против a[key] на общей сетке, в %."""
    xc = np.intersect1d(np.round(a[xkey], 6), np.round(b[xkey], 6))
    ia = np.searchsorted(np.round(a[xkey], 6), xc)
    ib = np.searchsorted(np.round(b[xkey], 6), xc)
    return xc, 100.0 * (b[key][ib] / a[key][ia] - 1.0)
