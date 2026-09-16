#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
qedfci_atom.py
==============
Поляризуемость АТОМА в оптической полости на уровне QED-FCI, включая
ВОЗБУЖДЁННЫЕ состояния. Скан по коэффициенту связи lambda при фиксированной
геометрии (у атома её и нет).

Мотивация: часовой переход 2^3S_1 -> 2^1S_0 гелия на 1557 нм измерен с
точностью 1.0e-12 в конденсате на МАГИЧЕСКОЙ длине волны. Магическая длина
волны определяется обращением в ноль ДИФФЕРЕНЦИАЛЬНОЙ динамической
поляризуемости, а BBR-сдвиг — дифференциальной статической. То есть
поляризуемости обоих часовых состояний и есть метрологически значимые
величины, а не побочные.

Контраст велик: alpha(1^1S_0) = 1.38 а.е., alpha(2^3S_1) = 315.63 а.е.
Дифференциальная поляризуемость почти целиком определяется возбуждённым
состоянием.

ПОЧЕМУ АТОМ УДОБЕН
------------------
Два электрона у He: FCI точен, заморозка остова не нужна, вопрос о качестве
референса не стоит. Атом нейтрален — дипольный оператор не зависит от начала
координат. Дипольный момент любого собственного состояния равен нулю, значит
когерентный сдвиг z = 0 и самосогласование не требуется.

Свободный атом сферически симметричен, alpha изотропна. Мода полости ломает
эту симметрию, и anisotropy alpha_zz != alpha_xx возникает ИЗ НИЧЕГО, только
из-за полости.

ВЫБОР СОСТОЯНИЯ
---------------
--ms задаётся как 2*M_S = N_alpha - N_beta (соглашение PySCF), поэтому для
двух электронов допустимы только чётные значения.

--ms 2            сектор M_S = 1, то есть (n_alpha, n_beta) = (2, 0).
                  Там живут ТОЛЬКО триплеты, и 2^3S — низший корень.
                  Ничего дополнительного не нужно.
--ms 0 --spin-filter --root 1
                  сектор M_S = 0 содержит и синглеты, и M_S=0-компоненты
                  триплетов: 1^1S, 2^3S, 2^1S. Гамильтониан спин-независим,
                  поэтому компоненты триплета строго вырождены с сектором
                  M_S=1 — решаем его отдельно и вычёркиваем совпадающие по
                  энергии корни. Остаются чистые синглеты, 2^1S идёт вторым.
--ms 0 --target-spin 0 --root 1
                  альтернатива через штраф shift*(S^2 - s(s+1)). Работает
                  только при norb < 64: в PySCF gen_des_str_index не
                  поддерживает большие пространства.

Примеры:
    # основное состояние 1^1S_0
    python qedfci_atom.py --atom He --ms 0 --root 0 \
        --lam-scan 0.0 0.01 0.05 0.1 --omega 0.0293

    # метастабильный триплет 2^3S_1
    python qedfci_atom.py --atom He --ms 2 --root 0 \
        --lam-scan 0.0 0.01 0.05 0.1 --omega 0.0293

    # синглет 2^1S_0 (второй синглет после основного)
    python qedfci_atom.py --atom He --ms 0 --spin-filter --root 1 \
        --lam-scan 0.0 0.01 0.05 0.1 --omega 0.0293
"""

import argparse
import os
import sys
import time

import numpy as np

from pyscf import gto, scf, ao2mo, lib
from pyscf.gto.basis import parse_gaussian
from pyscf.fci import cistring, direct_spin1, spin_op
from pyscf.fci.direct_spin1 import _unpack_nelec


AXIS = 'xyz'

ATOMS = {
    'He': dict(symbol='He', charge=0),
    'Be': dict(symbol='Be', charge=0),
    'Mg': dict(symbol='Mg', charge=0),
    'Ca': dict(symbol='Ca', charge=0),
}

COLUMNS = ['lambda', 'E', 'a_xx', 'a_yy', 'a_zz', 'a_iso', 'a_aniso',
           'err_xx', 'err_zz', 'dse_a_xx', 'dse_a_zz', 'dse_a_iso',
           'boson_gap', 'status']


def log(msg=''):
    print(msg, flush=True)


# ════════════════════════════════════════════════════════════════
#  Ядро электрон-бозонного FCI
# ════════════════════════════════════════════════════════════════
def fboson_ci_shape(norb, nelec, nmode=0, boson_states=None):
    if boson_states is not None:
        boson_states = (np.array([boson_states] * nmode)
                        if isinstance(boson_states, int)
                        else np.array(boson_states))
    else:
        boson_states = np.array([], dtype=int)
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    cishape = (na, nb)
    if nmode > 0:
        cishape = (na, nb) + tuple(boson_states + 1)
    return cishape


def _phot_view(ci0, cishape):
    na, nb = cishape[0], cishape[1]
    nph = int(np.prod(cishape[2:])) if len(cishape) > 2 else 1
    return ci0.reshape(na, nb, nph), nph


def _slices_for(imode, nmode, fock_id):
    sl = [slice(None)] * (2 + nmode)
    sl[2 + imode] = fock_id
    return tuple(sl)


def contract_2e_fast(eri, fcivec, norb, nelec, nmode, boson_states):
    """Электронная часть послойно через C-код PySCF."""
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    flat, nph = _phot_view(np.asarray(fcivec).reshape(cishape), cishape)
    out = np.empty_like(flat)
    for p in range(nph):
        out[:, :, p] = direct_spin1.contract_2e(
            eri, np.ascontiguousarray(flat[:, :, p]), norb, nelec)
    return out.reshape(np.asarray(fcivec).shape)


def contract_eb_fast(Heb, fcivec, norb, nelec, nmode, boson_states):
    """H_eb = sum_pq Heb_pq E_pq (b^+ + b): E_pq послойно, затем лестница."""
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = np.asarray(fcivec).reshape(cishape)
    fcinew = np.zeros(cishape)
    for imode in range(nmode):
        nboson = boson_states[imode]
        cre = np.sqrt(np.arange(1, nboson + 1))
        flat, nph = _phot_view(ci0, cishape)
        Ec_flat = np.empty_like(flat)
        for p in range(nph):
            Ec_flat[:, :, p] = direct_spin1.contract_1e(
                Heb[imode], np.ascontiguousarray(flat[:, :, p]), norb, nelec)
        Ec = Ec_flat.reshape(cishape)
        for ip in range(nboson):
            lo = _slices_for(imode, nmode, ip)
            hi = _slices_for(imode, nmode, ip + 1)
            fcinew[hi] += cre[ip] * Ec[lo]
            fcinew[lo] += cre[ip] * Ec[hi]
    return fcinew.reshape(np.asarray(fcivec).shape)


def contract_bb(Hbb, fcivec, norb, nelec, nmode, boson_states):
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = np.asarray(fcivec).reshape(cishape)
    fcinew = np.zeros(cishape)
    for imode in range(nmode):
        for i in range(1, boson_states[imode] + 1):
            s = _slices_for(imode, nmode, i)
            fcinew[s] += ci0[s] * (Hbb[imode, imode] * i)
    return fcinew.reshape(np.asarray(fcivec).shape)


def contract_ss_layers(fcivec, norb, nelec, nmode, boson_states):
    """S^2 действует только на электронные индексы — применяем послойно.

    ВНИМАНИЕ: pyscf.fci.spin_op.contract_ss опирается на
    cistring.gen_des_str_index, который не поддерживает norb >= 64.
    В большом диффузном базисе используйте --spin-filter: он отсеивает
    триплеты по вырождению с сектором M_S+1 и S^2 не вычисляет вовсе.
    """
    if norb >= 64:
        raise NotImplementedError(
            f'S^2 недоступен при norb = {norb} >= 64 (ограничение PySCF). '
            f'Используйте --spin-filter вместо --target-spin, либо уменьшите '
            f'базис, например --diffuse-lmax 1.')
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    flat, nph = _phot_view(np.asarray(fcivec).reshape(cishape), cishape)
    out = np.empty_like(flat)
    for p in range(nph):
        out[:, :, p] = spin_op.contract_ss(
            np.ascontiguousarray(flat[:, :, p]), norb, nelec)
    return out.reshape(np.asarray(fcivec).shape)


def absorb_h1e(h1e, eri, norb, nelec, fac=0.5):
    if not isinstance(nelec, (int, np.integer)):
        nelec = sum(nelec)
    h2e = np.array(eri, copy=True).reshape(norb, norb, norb, norb)
    f1e = h1e - np.einsum('jiik->jk', h2e) * 0.5
    f1e *= 1.0 / (nelec + 1e-100)
    for k in range(norb):
        h2e[k, k, :, :] += f1e
        h2e[:, :, k, k] += f1e
    return h2e * fac


def make_hdiag(h1e, eri, Hbb, norb, nelec, nmode, boson_states):
    neleca, nelecb = _unpack_nelec(nelec)
    ci_shape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    hdiag = np.zeros(ci_shape)
    occslista = cistring.gen_occslst(range(norb), neleca)
    occslistb = cistring.gen_occslst(range(norb), nelecb)
    eri4 = ao2mo.restore(1, eri, norb)
    diagj = np.einsum('iijj->ij', eri4)
    diagk = np.einsum('ijji->ij', eri4)
    for ia, aocc in enumerate(occslista):
        for ib, bocc in enumerate(occslistb):
            e1 = h1e[aocc, aocc].sum() + h1e[bocc, bocc].sum()
            e2 = (diagj[aocc][:, aocc].sum() + diagj[aocc][:, bocc].sum()
                  + diagj[bocc][:, aocc].sum() + diagj[bocc][:, bocc].sum()
                  - diagk[aocc][:, aocc].sum() - diagk[bocc][:, bocc].sum())
            hdiag[ia, ib] = e1 + 0.5 * e2
    for imode in range(nmode):
        for i in range(boson_states[imode] + 1):
            hdiag[_slices_for(imode, nmode, i)] += i * Hbb[imode, imode]
    return hdiag.ravel()


def qed_fci_kernel(H1, H2, norb, nelec, nmode, boson_states, Heb, Hbb,
                   ecore=0.0, tol=1e-11, max_cycle=300, max_space=24,
                   nroots=1, target_spin=None, spin_shift=1.0, init_ci=None):
    """-> (энергии, векторы). При target_spin не None добавляется штраф
    shift*(S^2 - s(s+1)), поднимающий состояния «чужой» мультиплетности."""
    if Hbb.ndim == 1:
        Hbb = np.diag(Hbb)
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    size = int(np.prod(cishape))
    H2mod = absorb_h1e(H1, H2, norb, nelec, 0.5)
    ss_target = None if target_spin is None else target_spin * (target_spin + 1)

    def hop(c):
        c = np.asarray(c).reshape(cishape)
        out = contract_2e_fast(H2mod, c, norb, nelec, nmode, boson_states)
        out = out + contract_eb_fast(Heb, c, norb, nelec, nmode, boson_states)
        out = out + contract_bb(Hbb, c, norb, nelec, nmode, boson_states)
        if ss_target is not None:
            ss = contract_ss_layers(c, norb, nelec, nmode, boson_states)
            out = out + spin_shift * (ss - ss_target * c)
        return np.asarray(out).reshape(-1)

    hdiag = make_hdiag(H1, H2, Hbb, norb, nelec, nmode, boson_states)
    precond = lib.make_diag_precond(hdiag, level_shift=1e-3)

    if init_ci is not None and len(init_ci) >= nroots:
        x0 = [np.asarray(v).reshape(-1).copy() for v in init_ci[:nroots]]
    else:
        idx = np.argsort(hdiag)[:max(nroots, 1)]
        x0 = []
        for i in idx:
            v = np.zeros(size)
            v[i] = 1.0
            x0.append(v)

    conv, e, c = lib.davidson1(lambda xs: [hop(x) for x in xs], x0, precond,
                               tol=tol, max_cycle=max_cycle,
                               max_space=max_space, nroots=nroots, verbose=0)
    e = np.atleast_1d(np.asarray(e)) + ecore
    c = [np.asarray(v).reshape(cishape) for v in np.atleast_2d(np.asarray(c))]
    return e, c, np.all(conv)


def spin_squared(civec, norb, nelec, nmode, boson_states):
    ss = contract_ss_layers(civec, norb, nelec, nmode, boson_states)
    return float(np.vdot(civec.ravel(), np.asarray(ss).ravel())
                 / np.vdot(civec.ravel(), civec.ravel()))


# ════════════════════════════════════════════════════════════════
#  Базис: диффузное расширение
# ════════════════════════════════════════════════════════════════
def augment_diffuse(bas, n_extra, ratio=3.0, lmax=2):
    r"""Четномасштабированное расширение: к каждому l добавляются n_extra
    несокращённых функций с экспонентами zeta_min / ratio^k.

    Для метастабильных состояний с большим <r> обычных aug-наборов не
    хватает: alpha(2^3S) = 315.63 а.е. против 1.38 у основного состояния,
    то есть плотность на порядок протяжённее. Диффузный хвост здесь важнее
    качества описания остова.
    """
    if n_extra <= 0:
        return bas
    by_l = {}
    for shell in bas:
        l = shell[0]
        by_l.setdefault(l, []).extend(prim[0] for prim in shell[1:])
    new = [list(s) for s in bas]
    for l in sorted(by_l):
        if l > lmax:
            continue
        zmin = min(by_l[l])
        for k in range(1, n_extra + 1):
            new.append([l, [zmin / ratio ** k, 1.0]])
    return new


def load_basis(spec, symbol, n_diffuse=0, ratio=3.0, lmax=2):
    path = spec
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), spec)
    if os.path.exists(path):
        bas = parse_gaussian.load(path, symbol)
    else:
        bas = gto.basis.load(spec, symbol)
        log(f'  базис {spec!r} из встроенной библиотеки PySCF')
    if n_diffuse:
        n0 = len(bas)
        bas = augment_diffuse(bas, n_diffuse, ratio, lmax)
        log(f'  добавлено {len(bas) - n0} диффузных оболочек '
            f'(отношение {ratio}, l <= {lmax})')
    return bas


# ════════════════════════════════════════════════════════════════
#  Гамильтониан
# ════════════════════════════════════════════════════════════════
def build_atom_data(symbol, charge, basis, spin, omega, mode_vec,
                    scf_conv=1e-12, nmo_max=None):
    mol = gto.Mole()
    mol.atom = f'{symbol} 0 0 0'
    mol.basis = {symbol: basis}
    mol.charge = charge
    mol.spin = spin
    mol.unit = 'Angstrom'
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.verbose = 0
    mf.conv_tol = scf_conv
    mf.max_cycle = 300
    mf.kernel()

    mo = mf.mo_coeff
    # Обрезание виртуального пространства по энергии орбитали.
    # Стоимость contract_2e идёт как norb^4, а для двухэлектронного
    # S-состояния alpha определяется возбуждениями s->p в ДИФФУЗНЫЕ
    # орбитали (низкие по энергии). Плотные виртуали с большими
    # экспонентами вклада почти не дают, но norb раздувают. Оставляем
    # nmo_max низших МО — это CAS, и сходимость по nmo_max надо проверять.
    if nmo_max is not None and nmo_max < mo.shape[1]:
        # Резать можно ТОЛЬКО по границам вырожденных групп. p-оболочка
        # трёхкратно вырождена, d — пятикратно; разрыв набора (скажем,
        # оставить p_x, p_y и выбросить p_z) ломает сферическую симметрию
        # атома, и alpha_xx перестаёт равняться alpha_zz. Проверка простая:
        # при lambda = 0 анизотропия обязана быть численным нулём.
        idx = np.argsort(mf.mo_energy)
        e_sorted = np.asarray(mf.mo_energy)[idx]
        k = int(nmo_max)
        while k < len(e_sorted) and abs(e_sorted[k] - e_sorted[k - 1]) < 1e-6:
            k += 1
        mo = mo[:, np.sort(idx[:k])]
    norb = mo.shape[1]
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo), norb)

    dip_ao = mol.intor('int1e_r', comp=3)      # атом в начале координат
    dip_mo = np.einsum('xpq,pi,qj->xij', dip_ao, mo, mo)
    D = np.einsum('x,xpq->pq', np.asarray(mode_vec, float), dip_mo)

    # DSE: 1e-часть 0.5*(D@D), 2e-часть с коэффициентом 1.0.
    # Ядерный диполь у атома в начале координат равен нулю, поэтому
    # перекрёстных и константных членов нет, и когерентный сдвиг z = 0.
    H1_base = h1e + 0.5 * (D @ D)
    H2 = eri + np.einsum('pq,rs->pqrs', D, D)
    Heb = (-np.sqrt(omega / 2.0) * D)[None, :, :]
    Hbb = np.array([[omega]])

    return dict(mol=mol, mf=mf, norb=norb, H1_base=H1_base, H2=H2,
                Heb=Heb, Hbb=Hbb, ecore=mol.energy_nuc(), dip_mo=dip_mo)


def field_terms(dat, field):
    """H' = -mu.F; для электронов +F.<r>. Ядро в начале координат — вклада нет."""
    F = np.asarray(field, float)
    return dat['H1_base'] + np.einsum('x,xij->ij', F, dat['dip_mo']), dat['ecore']


# ════════════════════════════════════════════════════════════════
#  Конечные разности
# ════════════════════════════════════════════════════════════════
def field_list(h, with_2h=True):
    e = np.eye(3)
    out = [('0', np.zeros(3))]
    for i in (2, 0):                     # z и x; y равен x по симметрии
        a = AXIS[i]
        out += [(f'+{a}', h * e[i]), (f'-{a}', -h * e[i])]
        if with_2h:
            out += [(f'+2{a}', 2 * h * e[i]), (f'-2{a}', -2 * h * e[i])]
    return out


def derive_props(E, h, with_2h=True):
    alpha = np.zeros(3)
    err = np.full(3, np.nan)
    E0 = E['0']
    for slot, i in ((2, 2), (0, 0)):
        a = AXIS[i]
        a_h = -(E[f'+{a}'] - 2 * E0 + E[f'-{a}']) / h**2
        if with_2h:
            a_2h = -(E[f'+2{a}'] - 2 * E0 + E[f'-2{a}']) / (2 * h)**2
            alpha[slot] = (4 * a_h - a_2h) / 3
            err[slot] = abs(a_h - a_2h)
        else:
            alpha[slot] = a_h
    alpha[1] = alpha[0]                  # a_yy = a_xx при моде вдоль z
    err[1] = err[0]
    return alpha, err


def photon_occ(civec, cishape):
    """<n_phot> для данного вектора.

    Нужно потому, что FCI-пространство содержит ФОТОННЫЕ РЕПЛИКИ каждого
    электронного состояния: при lambda = 0 секторы отщеплены и уровни идут
    как E_el + n*omega. Для He электронное возбуждение 2^1S лежит на 0.73 Ha
    выше основного, а реплики — через omega = 0.029, так что между ними их
    десятки. Отбирать состояние по номеру корня бессмысленно; физическому
    состоянию отвечает корень с <n> ~ 0.
    """
    flat, nph = _phot_view(np.asarray(civec), cishape)
    w = np.einsum('abp,abp->p', flat, flat)
    tot = w.sum()
    if tot <= 0:
        return float('nan')
    return float(np.dot(np.arange(nph), w) / tot)


def solve_state(dat, H1, ecore, norb, nelec, args, nroots, init_ci=None,
                ref_vec=None, verbose=False):
    """Решает и отбирает нужное состояние.

    Отбор в три этапа, каждый следующий надёжнее предыдущего:

      1) по заселённости фотонов <n> < --max-photon-occ — выбрасывает
         фотонные реплики электронных состояний;
      2) по <S^2>, если задан --require-spin — при lambda != 0 порядок
         уровней меняется, и отбор по номеру начинает брать состояние
         чужой мультиплетности;
      3) по ПЕРЕКРЫТИЮ с опорным вектором предыдущей точки, если он дан.
         Это главный механизм: индекс корня не сохраняется при включении
         поля и связи, а перекрытие сохраняется.

    Если опорного вектора нет (самая первая точка), берётся корень --root
    среди прошедших фильтры.
    """
    kw = dict(ecore=ecore, tol=args.fci_tol, max_cycle=args.fci_cycles,
              max_space=args.max_space, target_spin=args.target_spin,
              spin_shift=args.spin_shift)
    cishape = fboson_ci_shape(norb, nelec, 1, [args.nboson])

    e_all, c_all, conv = qed_fci_kernel(H1, dat['H2'], norb, nelec, 1,
                                        [args.nboson], dat['Heb'], dat['Hbb'],
                                        nroots=nroots, init_ci=init_ci, **kw)
    e_all = np.atleast_1d(e_all)
    occ = [photon_occ(ci, cishape) for ci in c_all]

    s2 = [None] * len(c_all)
    if args.require_spin is not None and norb < 64:
        s2 = [spin_squared(ci, norb, nelec, 1, [args.nboson]) for ci in c_all]
    ss_want = (None if args.require_spin is None
               else args.require_spin * (args.require_spin + 1))

    e_hi = None
    na, nb = nelec
    if args.spin_filter and nb >= 1:
        e_hi, _, _ = qed_fci_kernel(H1, dat['H2'], norb, (na + 1, nb - 1), 1,
                                    [args.nboson], dat['Heb'], dat['Hbb'],
                                    nroots=nroots, **kw)
        e_hi = np.atleast_1d(e_hi)

    keep = []
    for i, ei in enumerate(e_all):
        if occ[i] > args.max_photon_occ:
            continue
        if ss_want is not None and s2[i] is not None \
                and abs(s2[i] - ss_want) > args.spin_tol:
            continue
        if e_hi is not None and np.min(np.abs(e_hi - ei)) < args.spin_filter_tol:
            continue
        keep.append(i)

    if not keep:
        raise RuntimeError('ни один корень не прошёл отбор; ослабьте '
                           '--max-photon-occ / --spin-tol или увеличьте --nroots')

    # отслеживание по перекрытию
    ovl = None
    if ref_vec is not None:
        r = np.asarray(ref_vec).ravel()
        rn = np.linalg.norm(r)
        ovl = []
        for i in keep:
            v = np.asarray(c_all[i]).ravel()
            ovl.append(abs(np.dot(r, v)) / (rn * np.linalg.norm(v)))
        idx = keep[int(np.argmax(ovl))]
        best = max(ovl)
    else:
        if len(keep) <= args.root:
            raise RuntimeError(
                f'после отбора осталось {len(keep)} корней, запрошен '
                f'№{args.root}; увеличьте --nroots')
        idx = keep[args.root]
        best = None

    if verbose:
        log('      корни (E, <n_phot>, <S^2>, перекрытие, отбор):')
        for i, ei in enumerate(e_all):
            tag = 'ВЗЯТ' if i == idx else ('годен' if i in keep else 'отброшен')
            s2s = f'{s2[i]:5.2f}' if s2[i] is not None else '  -- '
            os_ = ''
            if ovl is not None and i in keep:
                os_ = f'  ovl = {ovl[keep.index(i)]:.4f}'
            log(f'        {i:2d}  {ei: .8f}   <n> = {occ[i]:5.3f}   '
                f'<S^2> = {s2s}{os_}   {tag}')
        if best is not None and best < args.overlap_warn:
            log(f'      !! максимальное перекрытие {best:.3f} < '
                f'{args.overlap_warn}: состояние сильно перемешано, '
                f'понятие «поляризуемость этого состояния» под вопросом')

    return e_all[idx:idx + 1], [c_all[idx]], conv


# ════════════════════════════════════════════════════════════════
#  Аргументы
# ════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--atom', default='He', choices=sorted(ATOMS))
    p.add_argument('--charge', type=int, default=None)
    p.add_argument('--lam-scan', type=float, nargs='+',
                   default=[0.0, 0.01, 0.025, 0.05, 0.1], metavar='L')
    p.add_argument('--omega', type=float, default=0.0293,
                   help='частота моды, а.е. (0.0293 = 1557 нм, часовой '
                        'переход He; 0.0421 = 1083 нм, 2^3S->2^3P)')
    p.add_argument('--mode-dir', type=float, nargs=3, default=[0., 0., 1.])

    p.add_argument('--basis-file', default='aug-cc-pvqz',
                   help='путь к .gbs или имя встроенного базиса PySCF')
    p.add_argument('--basis-label', default=None)
    p.add_argument('--diffuse', type=int, default=0, metavar='N',
                   help='добавить N четномасштабированных диффузных функций '
                        'на каждое l (нужно для метастабильных состояний)')
    p.add_argument('--diffuse-ratio', type=float, default=3.0)
    p.add_argument('--diffuse-lmax', type=int, default=2)
    p.add_argument('--nmo-max', type=int, default=None, metavar='N',
                   help='оставить только N низших по энергии МО. Стоимость '
                        'падает как (N/norb)^4, а для S-состояний плотные '
                        'виртуали в alpha почти не вносят. Сходимость по N '
                        'обязательно проверять')

    p.add_argument('--ms', type=int, default=0,
                   help='2*M_S: 0 -> (n_a,n_b)=(N/2,N/2), 1 -> сдвиг на 1 '
                        'электрон (для He: 1 даёт сектор триплета)')
    p.add_argument('--root', type=int, default=0,
                   help='какой корень брать (0 = низший)')
    p.add_argument('--nroots', type=int, default=None,
                   help='сколько корней искать (по умолчанию root+1)')
    p.add_argument('--target-s2', type=float, default=None, metavar='S2',
                   help='отбирать корни по <S^2>: 0 для синглета, 2 для '
                        'триплета. Надёжнее отбора по индексу, который '
                        'ломается при перестройке уровней с ростом lambda')
    p.add_argument('--s2-tol', type=float, default=0.1)
    p.add_argument('--require-spin', type=float, default=None, metavar='S',
                   help='жёсткий фильтр по <S^2>: оставлять только корни с '
                        'S(S+1). Для 2^1S нужно --require-spin 0, для 2^3S '
                        '--require-spin 1. Работает при norb < 64')
    p.add_argument('--spin-tol', type=float, default=0.1)
    p.add_argument('--overlap-warn', type=float, default=0.7,
                   help='порог перекрытия, ниже которого печатается '
                        'предупреждение о сильном перемешивании состояния')
    p.add_argument('--max-photon-occ', type=float, default=0.5, metavar='N',
                   help='порог по <n_phot> при отборе корней: фотонные '
                        'реплики электронных состояний отбрасываются')
    p.add_argument('--spin-filter', action='store_true',
                   help='отсеивать триплеты по вырождению с сектором M_S+1: '
                        'гамильтониан спин-независим, поэтому компоненты '
                        'триплета строго вырождены. Работает при любом norb, '
                        'в отличие от --target-spin')
    p.add_argument('--spin-filter-tol', type=float, default=1e-6,
                   help='допуск при сопоставлении энергий, Ha')
    p.add_argument('--target-spin', type=float, default=None, metavar='S',
                   help='спиновый штраф на S: поднимает состояния другой '
                        'мультиплетности. Для 2^1S: --ms 0 --target-spin 0')
    p.add_argument('--spin-shift', type=float, default=1.0)

    p.add_argument('--ff-step', type=float, default=0.002)
    p.add_argument('--no-richardson', action='store_true')
    p.add_argument('--nboson', type=int, default=4)
    p.add_argument('--boson-check-list', type=int, nargs='+', default=None)
    p.add_argument('--no-dse-only', action='store_true')

    p.add_argument('--fci-tol', type=float, default=1e-11)
    p.add_argument('--fci-cycles', type=int, default=400)
    p.add_argument('--max-space', type=int, default=24)
    p.add_argument('--scf-conv', type=float, default=1e-12)
    p.add_argument('--outdir', default='.')
    return p.parse_args(argv)


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════
def main(argv=None):
    args = parse_args(argv)
    cfg = ATOMS[args.atom]
    charge = cfg['charge'] if args.charge is None else args.charge
    label = args.basis_label or args.basis_file.replace('/', '_')
    if args.diffuse:
        label += f'+{args.diffuse}d'
    nroots = args.nroots if args.nroots else args.root + 1

    mode_dir = np.asarray(args.mode_dir, float)
    mode_dir = mode_dir / np.linalg.norm(mode_dir)

    basis = load_basis(args.basis_file, cfg['symbol'], args.diffuse,
                       args.diffuse_ratio, args.diffuse_lmax)

    os.makedirs(args.outdir, exist_ok=True)
    tag = (f'{args.atom}_ms{args.ms}_root{args.root}'
           + (f'_S{args.target_spin:g}' if args.target_spin is not None else '')
           + f'_om{args.omega:.4f}_{label}')
    datfile = os.path.join(args.outdir, f'{tag}_data.txt')

    log('\n' + '#' * 92)
    log(f'#  {args.atom}   QED-FCI   omega = {args.omega}   базис {label}')
    log(f'#  сектор M_S: --ms {args.ms};  корень {args.root} из {nroots}'
        + (f';  спиновый штраф на S = {args.target_spin:g}'
           if args.target_spin is not None else ''))
    log(f'#  скан по lambda: {args.lam_scan}')
    log(f'#  h = {args.ff_step};  фотонов до {args.nboson}')
    log('#' * 92)
    log('#  Стоимость contract_2e растёт как norb^4, а размер FCI-пространства')
    log('#  для двух электронов мал. Поэтому решает именно число орбиталей:')
    log('#  --diffuse-lmax 1 (только s,p) и меньший базовый набор дают')
    log('#  кратное ускорение почти без потери точности для S-состояний.')

    fh = open(datfile, 'w')
    fh.write(f'# atom = {args.atom}, charge = {charge}, ms = {args.ms}, '
             f'root = {args.root}, target_spin = {args.target_spin}\n')
    fh.write(f'# omega = {args.omega}, basis = {label}, '
             f'diffuse = {args.diffuse}, h = {args.ff_step}, '
             f'nboson = {args.nboson}\n')
    fh.write('# columns          = ' + ','.join(COLUMNS) + '\n')
    fh.flush()

    # --ms = 2*M_S = N_alpha - N_beta; должно совпадать по чётности с N
    probe = gto.M(atom=f'{cfg["symbol"]} 0 0 0', basis={cfg['symbol']: basis},
                  charge=charge, verbose=0)
    ne_probe = probe.nelectron
    if (ne_probe - args.ms) % 2 or args.ms < 0 or args.ms > ne_probe:
        log(f'\n  ОШИБКА: --ms {args.ms} несовместимо с {ne_probe} электронами.')
        log(f'  --ms это 2*M_S = N_alpha - N_beta, то есть допустимы '
            f'{[m for m in range(0, ne_probe + 1) if (ne_probe - m) % 2 == 0]}.')
        log(f'  Для триплета He (M_S = 1) нужно --ms 2, а не --ms 1.')
        return 1

    pts = field_list(args.ff_step, not args.no_richardson)
    t0 = time.time()
    ref_lambda = None            # опорный вектор с предыдущего значения lambda

    for lam in args.lam_scan:
        mode_vec = lam * mode_dir
        log(f'\n  lambda = {lam:.4f}   [прошло {time.time() - t0:6.0f} s]')

        # spin: число неспаренных электронов в секторе
        spin_scf = args.ms
        dat = build_atom_data(cfg['symbol'], charge, basis, spin_scf,
                              args.omega, mode_vec, args.scf_conv,
                              nmo_max=args.nmo_max)
        norb = dat['norb']
        ne_tot = dat['mol'].nelectron
        na = (ne_tot + args.ms) // 2
        nb = ne_tot - na
        nelec = (na, nb)
        log(f'    norb = {norb}'
            + (f' (обрезано из {dat["mf"].mo_coeff.shape[1]})'
               if args.nmo_max else '')
            + f',  (n_alpha, n_beta) = {nelec}')

        en, en_dse = {}, {}
        ci_prev = None
        ok = True
        ref_here = None          # опора внутри одной lambda (между точками поля)
        for tag_f, F in pts:
            H1, ecore = field_terms(dat, F)
            # на нулевом поле опираемся на состояние с предыдущей lambda,
            # дальше — на состояние при F = 0 этой же lambda
            ref = ref_here if tag_f != '0' else ref_lambda
            e, c, conv = solve_state(dat, H1, ecore, norb, nelec, args,
                                     nroots, init_ci=ci_prev, ref_vec=ref,
                                     verbose=(tag_f == '0'))
            ci_prev = None
            if tag_f == '0':
                ref_here = np.asarray(c[0]).copy()
                ref_lambda = ref_here.copy()
            if not conv:
                log(f'      [{tag_f:>4s}] Дэвидсон не сошёлся')
                ok = False
            en[tag_f] = float(e[0])          # solve_state уже отобрал нужный
            if tag_f == '0':
                try:
                    s2 = spin_squared(c[0], norb, nelec, 1,
                                      [args.nboson])
                    s2s = f'<S^2> = {s2:.4f}'
                except NotImplementedError:
                    s2s = '<S^2> недоступно (norb >= 64)'
                log(f'      E = {en[tag_f]: .10f}   {s2s}')


            # при lambda = 0 связи нет, предел omega->0 тождественно
            # совпадает с полным расчётом — не тратим время
            if not args.no_dse_only and abs(lam) > 1e-12:
                dat0 = dict(dat)
                dat0['Heb'] = dat['Heb'] * 0.0
                e2, _, _ = solve_state(dat0, H1, ecore, norb, nelec, args,
                                       nroots, ref_vec=ref)
                en_dse[tag_f] = float(e2[0])

        if not ok:
            fh.write(f'{lam:13.6f}  FAILED\n'); fh.flush(); continue

        alpha, err = derive_props(en, args.ff_step, not args.no_richardson)
        aiso = float(np.mean(alpha))
        aani = float(alpha[2] - alpha[0])
        if en_dse:
            adse, _ = derive_props(en_dse, args.ff_step, not args.no_richardson)
        else:
            adse = np.full(3, np.nan)

        log(f'      alpha:  xx = {alpha[0]:10.4f}   zz = {alpha[2]:10.4f}   '
            f'iso = {aiso:10.4f}   anis = {aani:9.4f}')
        if en_dse:
            log(f'      предел omega->0:  xx = {adse[0]:10.4f}   '
                f'zz = {adse[2]:10.4f}   фотонный вклад: '
                f'xx {alpha[0] - adse[0]:+8.4f}  zz {alpha[2] - adse[2]:+8.4f}')
        log(f'      err:  xx {err[0]:.2e}   zz {err[2]:.2e}')

        status = 'OK'
        flags = []
        if np.any(alpha <= 0):
            flags.append('NEG_ALPHA')
        # при lambda = 0 атом сферически симметричен: анизотропия обязана
        # быть нулём. Ненулевая означает разрыв вырожденной оболочки при
        # --nmo-max либо иную поломку симметрии.
        if abs(lam) < 1e-12 and abs(aani) > 1e-3 * max(abs(aiso), 1.0):
            flags.append(f'ANISO_AT_ZERO={aani:.2e}')
            log(f'      !! при lambda=0 анизотропия должна быть нулём, '
                f'а вышла {aani:.4f}. Скорее всего --nmo-max разорвал '
                f'вырожденную оболочку.')
        if err[0] > 5e-3 * abs(alpha[0]):
            flags.append('STEP_XX')
        if flags:
            status = '+'.join(flags)
        vals = [lam, en['0'], alpha[0], alpha[1], alpha[2], aiso, aani,
                err[0], err[2], adse[0], adse[2], float(np.mean(adse)),
                float('nan')]
        fh.write('  '.join(f'{v:13.6f}' if abs(v) < 1e5 else f'{v:13.4e}'
                           for v in vals) + f'  {status}\n')
        fh.flush()

    fh.close()
    log(f'\nГотово за {time.time() - t0:.0f} s')
    log(f'  данные: {datfile}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
