#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
test_frozen_core.py
===================
Оценка ошибки замораживания остова для поляризуемости — БЕЗ полости
(lambda = 0), только PySCF. Нужен, чтобы решить, можно ли считать LiH на
QED-FCI в валентном пространстве: полный FCI для LiH/Sadlej даёт 279 тыс.
детерминантов, и промежуточный тензор (norb,norb)+cishape в питоновском
электрон-бозонном ядре занял бы ~16 ГБ. С замороженным 1s лития остаётся
два валентных электрона, ~1 тыс. детерминантов и 50 МБ.

Штатный FCI в PySCF написан на C, поэтому полный расчёт здесь быстрый —
он и служит эталоном.

Сравниваются три уровня при одинаковых орбиталях и одинаковой схеме
конечного поля:
    RHF        — среднее поле
    CASCI      — FCI в валентном пространстве (остов заморожен)
    FCI        — все электроны активны (эталон)

Схема поля та же, что в рабочих скриптах: 0, ±h, ±2h по каждой оси,
alpha_ii с экстраполяцией Ричардсона.

Геометрия центрируется на центр ядерного заряда, поэтому mu_nuc = 0 и
ядерный член внешнего поля обращается в ноль.

Запуск:
    python test_frozen_core.py --mol LiH --basis-file sadlej.gbs
    python test_frozen_core.py --mol LiH --basis-file sadlej.gbs -R 1.6 2.5 3.2
    python test_frozen_core.py --mol LiH --basis-file cc-pvdz --frozen 1 2
"""

import argparse
import os
import sys
import time

import numpy as np

from pyscf import gto, scf, ao2mo, mcscf, cc
from pyscf.fci import direct_spin1
from pyscf.gto.basis import parse_gaussian


MOLECULES = {
    'LiH': dict(template='Li 0 0 0; H 0 0 {R:.6f}', elements=('Li', 'H'),
                charge=0, spin=0, Re=1.5957, ncore=1),
    'HF':  dict(template='F 0 0 0; H 0 0 {R:.6f}',  elements=('F', 'H'),
                charge=0, spin=0, Re=0.9168, ncore=1),
    'H2':  dict(template='H 0 0 0; H 0 0 {R:.6f}',  elements=('H',),
                charge=0, spin=0, Re=0.7414, ncore=0),
}


def log(m=''):
    print(m, flush=True)


def load_basis(spec, elements):
    path = spec
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), spec)
    if os.path.exists(path):
        return {el: parse_gaussian.load(path, el) for el in elements}
    return {el: gto.basis.load(spec, el) for el in elements}


def make_mol(cfg, basis, R):
    mol = gto.Mole()
    mol.atom = cfg['template'].format(R=R)
    mol.basis = basis
    mol.charge = cfg['charge']
    mol.spin = cfg['spin']
    mol.unit = 'Angstrom'
    mol.verbose = 0
    mol.build()
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    zcen = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mol.set_geom_(coords - zcen, unit='Bohr')
    return mol


def hcore_with_field(mol, field):
    """H' = -mu.F; электронная часть +F.sum_i r_i."""
    h = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    with mol.with_common_orig((0., 0., 0.)):
        ao_r = mol.intor_symmetric('int1e_r', comp=3)
    return h + np.einsum('x,xij->ij', np.asarray(field, float), ao_r)


def energies_at_field(mol, field, frozen_list, conv=1e-12, do_ccsd=False):
    """-> dict: 'RHF', 'FCI', 'CAS{n}' для каждого n из frozen_list."""
    h_field = hcore_with_field(mol, field)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = conv
    mf.max_cycle = 300
    mf.get_hcore = lambda *a, **k: h_field
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f'RHF не сошёлся, field={field}')

    out = {'RHF': float(mf.e_tot)}
    norb = mf.mo_coeff.shape[1]

    # полный FCI (эталон)
    h1 = mf.mo_coeff.T @ h_field @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    e, _ = direct_spin1.kernel(h1, eri, norb, mol.nelec,
                               ecore=mol.energy_nuc(), verbose=0, tol=1e-11)
    out['FCI'] = float(np.ravel(e)[0])

    # CASCI с замороженным остовом; get_h1eff берёт _scf.get_hcore(),
    # то есть подменённый гамильтониан с полем
    for nfr in frozen_list:
        ncas = norb - nfr
        nelecas = mol.nelectron - 2 * nfr
        if nelecas < 2:
            continue
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver.conv_tol = 1e-11
        mc.kernel()
        out[f'CAS{nfr}'] = float(mc.e_tot)

    if do_ccsd:
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.conv_tol = 1e-11
        mycc.kernel()
        out['CCSD'] = float(mycc.e_tot)

    return out


def props_at_R(mol, frozen_list, h, do_ccsd=False):
    """mu_z, alpha_xx, alpha_zz для каждого уровня теории."""
    ez = np.array([0., 0., 1.])
    ex = np.array([1., 0., 0.])
    pts = {'0': np.zeros(3),
           '+z': h * ez, '-z': -h * ez, '+2z': 2 * h * ez, '-2z': -2 * h * ez,
           '+x': h * ex, '-x': -h * ex, '+2x': 2 * h * ex, '-2x': -2 * h * ex}

    raw = {}
    for tag, f in pts.items():
        t0 = time.time()
        raw[tag] = energies_at_field(mol, f, frozen_list, do_ccsd=do_ccsd)
        log(f'      [{tag:>4s}]  {time.time() - t0:6.1f} s')

    levels = list(raw['0'].keys())
    res = {}
    for lv in levels:
        E = {t: raw[t][lv] for t in pts}
        mu_z = -(-E['+2z'] + 8 * E['+z'] - 8 * E['-z'] + E['-2z']) / (12 * h)

        def rich(p, p2, m, m2):
            a_h = -(p - 2 * E['0'] + m) / h**2
            a_2h = -(p2 - 2 * E['0'] + m2) / (2 * h)**2
            return (4 * a_h - a_2h) / 3

        azz = rich(E['+z'], E['+2z'], E['-z'], E['-2z'])
        axx = rich(E['+x'], E['+2x'], E['-x'], E['-2x'])
        res[lv] = dict(E0=E['0'], mu_z=mu_z, axx=axx, azz=azz,
                       aiso=(2 * axx + azz) / 3)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mol', default='LiH', choices=sorted(MOLECULES))
    p.add_argument('--basis-file', default='sadlej.gbs')
    p.add_argument('-R', '--rvals', type=float, nargs='+', default=None,
                   help='список R, Å (по умолчанию R_e и R_e+1)')
    p.add_argument('--frozen', type=int, nargs='+', default=None,
                   help='сколько орбиталей замораживать (по умолчанию ncore)')
    p.add_argument('--ff-step', type=float, default=0.004)
    p.add_argument('--ccsd', action='store_true',
                   help='добавить CCSD для сравнения')
    args = p.parse_args(argv)

    cfg = MOLECULES[args.mol]
    basis = load_basis(args.basis_file, cfg['elements'])
    rvals = args.rvals if args.rvals else [cfg['Re'], cfg['Re'] + 1.0]
    frozen_list = args.frozen if args.frozen else [cfg['ncore']]
    frozen_list = [n for n in frozen_list if n > 0]

    # проверка на вырожденность: при nelecas < 2 CASCI сводится к RHF и
    # «ошибка заморозки» окажется на самом деле ошибкой среднего поля
    mol_probe = make_mol(cfg, load_basis(args.basis_file, cfg['elements']),
                         cfg['Re'])
    nmax = (mol_probe.nelectron - 2) // 2
    bad = [n for n in frozen_list if mol_probe.nelectron - 2 * n < 2]
    if bad:
        log(f'  !! заморозка {bad} оставляет меньше 2 электронов в активном '
            f'пространстве; для {args.mol} максимум {nmax}. Пропускаю.')
        frozen_list = [n for n in frozen_list if n not in bad]
    if not frozen_list:
        log('  нечего проверять'); return 1

    log('\n' + '=' * 88)
    log(f'  {args.mol}   базис {args.basis_file}   h = {args.ff_step}   '
        f'заморозка: {frozen_list}')
    log('  Вопрос: насколько CASCI с замороженным остовом отличается от '
        'полного FCI по alpha?')
    log('=' * 88)

    for R in rvals:
        mol = make_mol(cfg, basis, R)
        nao = mol.nao_nr()
        log(f'\n  R = {R:.4f} Å   (norb = {nao}, электронов = {mol.nelectron})')
        try:
            res = props_at_R(mol, frozen_list, args.ff_step, do_ccsd=args.ccsd)
        except Exception as exc:
            log(f'    СБОЙ: {exc}')
            continue

        ref = res['FCI']
        log(f'\n    {"уровень":>8s} {"E0":>16s} {"mu_z":>10s} {"a_xx":>10s} '
            f'{"a_zz":>10s} {"a_iso":>10s} {"откл. a_iso":>12s}')
        order = ['RHF'] + (['CCSD'] if args.ccsd else []) \
            + [f'CAS{n}' for n in frozen_list] + ['FCI']
        for lv in order:
            if lv not in res:
                continue
            r = res[lv]
            d = 100 * (r['aiso'] / ref['aiso'] - 1)
            log(f'    {lv:>8s} {r["E0"]:16.10f} {r["mu_z"]:10.5f} '
                f'{r["axx"]:10.4f} {r["azz"]:10.4f} {r["aiso"]:10.4f} '
                f'{d:11.3f}%')

        for n in frozen_list:
            lv = f'CAS{n}'
            if lv not in res:
                continue
            log(f'\n    заморозка {n} орб.:  '
                f'd(a_xx) = {100 * (res[lv]["axx"] / ref["axx"] - 1):+.3f}%   '
                f'd(a_zz) = {100 * (res[lv]["azz"] / ref["azz"] - 1):+.3f}%   '
                f'd(mu_z) = {100 * (res[lv]["mu_z"] / ref["mu_z"] - 1):+.3f}%')

    log('\n' + '=' * 88)
    log('  Как читать: если отклонение CASCI от FCI по alpha меньше ~0.5%,')
    log('  замораживание остова в qedfci_scan.py оправдано — эффект полости')
    log('  измеряется на уровне процентов, и такая систематика в разности')
    log('  alpha(lambda) - alpha(0) сократится почти полностью.')
    log('=' * 88)
    return 0


if __name__ == '__main__':
    sys.exit(main())
