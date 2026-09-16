#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
qedfci_scan.py
==============
Дипольный момент и тензор поляризуемости двухатомной молекулы в оптической
полости на уровне QED-FCI. Интерфейс тот же, что у qedcc_scan.py, чтобы
результаты двух методов сравнивались точка в точку.

Ядро расчёта (fboson_ci_shape, contract_2e, contract_eb, contract_bb,
absorb_h1e, make_hdiag, qed_fci_kernel, build_geometry_data) перенесено из
проверенного fci.py: оно сверено с пакетом eT и не ломается при сильной
связи и больших R.

ЕДИНСТВЕННОЕ ОТЛИЧИЕ ОТ fci.py ПО ФИЗИКЕ — знак электронного полевого члена.
В fci.py стояло  H1 -= F.<r>,  здесь  H1 += F.<r>.  Физическое возмущение
H' = -mu.F при mu_e = -sum_i r_i даёт именно +F.<r>, и тогда

    mu_i = -dE/dF_i,   alpha_ij = -d^2E/dF_i dF_j

работают напрямую и для полярных молекул тоже. В fci.py знак компенсировался
формулой mu_full = mu_nuc - mu_e_z, что верно лишь при mu_nuc = 0, то есть
только для центрированной геометрии. На alpha (чётная производная) смена
знака не влияет вовсе.

ПОЧЕМУ FCI, А НЕ CC
-------------------
QED-CCSD-U22-S2 систематически недооценивает электрон-фотонную корреляцию
из-за усечения по фотонным возбуждениям; для H2 это показано на рис. 6a в
J. Chem. Theory Comput. 21, 10035 (2025). Замер на H2/Sadlej при lambda=0.1,
R=0.74 дал расхождение по эффекту полости в 2.5 раза. QED-FCI от этого
свободен: число фоковских состояний задаётся --nboson и стоит линейно.

ПОСТРОЕНИЕ ГАМИЛЬТОНИАНА
------------------------
D_pq = (lambda . <p|r|q>) в МО. Разбиение DSE-члена АСИММЕТРИЧНО:

    H1 += 0.5 * (D @ D)
    H2 += 1.0 * outer(D, D)      <- именно 1.0, не 0.5

Коэффициент 1.0 возникает потому, что absorb_h1e складывает одноэлектронную
часть в ОБЕ пары индексов H2 (h2e[k,k,:,:] и h2e[:,:,k,k]); наивно
симметричное разбиение 0.5/0.5 даёт молча неверную энергию. Проверено
против прямой тензорной диагонализации до 1e-15 Ha.

Ядерный диполь входит как -mu_nuc_lam*D в H1, как
+sqrt(w/2)*(mu_nuc_lam/nelec)*I в Heb (поскольку sum_pq delta_pq E_pq = N_el)
и как +0.5*mu_nuc_lam^2 в ecore. При центрировании mu_nuc_lam = 0.

СХЕМЫ КОНЕЧНЫХ РАЗНОСТЕЙ
------------------------
    --mode-dir 0 0 1   вдоль оси     ->  9 точек поля (7 с --fast)
    --mode-dir 1 0 0   поперёк оси   -> 13 точек
    --mode-dir 1 0 1   наклон в xz   -> 17 точек
    --mode-dir 1 1 1   общий случай  -> 25 точек

Примеры:
    python qedfci_scan.py --mol H2 --lam 0.00 --omega 0.4687 \
        --rmin 0.74 --rmax 0.74 --verify
    python qedfci_scan.py --mol H2 --lam 0.10 --omega 0.4687 \
        --rmin 0.5 --rmax 4.0
"""

import argparse
import os
import sys
import time

import numpy as np

from pyscf import gto, scf, ao2mo, lib
from pyscf.gto.basis import parse_gaussian
from pyscf.fci import cistring, direct_spin1
from pyscf.fci.direct_spin1 import _unpack_nelec


AXIS = 'xyz'

MOLECULES = {
    'H2':   dict(template='H 0 0 0; H 0 0 {R:.6f}',   elements=('H',),
                 charge=0, spin=0, Re=0.7414),
    'LiH':  dict(template='Li 0 0 0; H 0 0 {R:.6f}',  elements=('Li', 'H'),
                 charge=0, spin=0, Re=1.5957),
    'HF':   dict(template='F 0 0 0; H 0 0 {R:.6f}',   elements=('F', 'H'),
                 charge=0, spin=0, Re=0.9168),
    'HeH+': dict(template='He 0 0 0; H 0 0 {R:.6f}',  elements=('He', 'H'),
                 charge=1, spin=0, Re=0.7743),
}

COLUMNS = ['R',
           'mu_abs', 'mu_x', 'mu_y', 'mu_z',
           'a_xx', 'a_yy', 'a_zz', 'a_xz', 'a_iso',
           'err_xx', 'err_yy', 'err_zz',
           'dse_mu_z', 'dse_a_xx', 'dse_a_yy', 'dse_a_zz', 'dse_a_iso',
           'boson_gap', 'status']


# ════════════════════════════════════════════════════════════════
#  Ядро электрон-бозонного FCI (из fci.py, без изменений)
# ════════════════════════════════════════════════════════════════
def fboson_ci_shape(norb, nelec, nmode=0, boson_states=None):
    if boson_states is not None:
        boson_states = (np.array([boson_states] * nmode)
                        if isinstance(boson_states, int)
                        else np.array(boson_states))
    else:
        boson_states = np.array([], dtype=int)
        if nmode > 0:
            raise TypeError("boson_states must be set if nmode > 0")
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    cishape = (na, nb)
    if nmode > 0:
        cishape = (na, nb) + tuple(boson_states + 1)
    return cishape


def contract_2e(eri, fcivec, norb, nelec, nmode, boson_states):
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)

    t1a = np.zeros((norb, norb) + cishape)
    t1b = np.zeros((norb, norb) + cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1a[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1b[a, i, :, str1] += sign * ci0[:, str0]

    t1 = t1a + t1b
    t2 = np.dot(eri.reshape(norb ** 2, -1), t1.reshape(norb ** 2, -1))
    t2 = t2.reshape((norb, norb) + cishape)

    fcinew = np.zeros(cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t2[a, i, str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += sign * t2[a, i, :, str0]
    return fcinew.reshape(fcivec.shape)


def _phot_view(ci0, cishape):
    """(na, nb, ...фотонные оси...) -> (na, nb, nph) без копирования."""
    na, nb = cishape[0], cishape[1]
    nph = int(np.prod(cishape[2:])) if len(cishape) > 2 else 1
    return ci0.reshape(na, nb, nph), nph


def contract_2e_fast(eri, fcivec, norb, nelec, nmode, boson_states):
    """Электронная часть послойно через C-код PySCF.

    Двухэлектронный оператор действует только на электронные индексы, то
    есть внутри каждого фотонного слоя независимо. Питоновская версия
    contract_2e строит промежуточный тензор формы (norb,norb)+cishape —
    для LiH с 32 активными орбиталями это 59 МБ на массив и ~1 с на
    итерацию Дэвидсона. Здесь тот же результат получается вызовом
    direct_spin1.contract_2e для каждого слоя.
    """
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = np.asarray(fcivec).reshape(cishape)
    flat, nph = _phot_view(ci0, cishape)
    out = np.empty_like(flat)
    for p in range(nph):
        out[:, :, p] = direct_spin1.contract_2e(
            eri, np.ascontiguousarray(flat[:, :, p]), norb, nelec)
    return out.reshape(fcivec.shape)


def contract_eb_fast(Heb, fcivec, norb, nelec, nmode, boson_states):
    """То же для электрон-фотонного члена: E_pq послойно, затем лестница."""
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
            s_low = _slices_for(imode, nmode, ip)
            s_high = _slices_for_cre(imode, nmode, ip)
            fcinew[s_high] += cre[ip] * Ec[s_low]
            fcinew[s_low] += cre[ip] * Ec[s_high]

    return fcinew.reshape(fcivec.shape)


def contract_all(H1, H2, Heb, Hbb, cin, norb, nelec, nmode, boson_states,
                 fast=True):
    c2 = contract_2e_fast if fast else contract_2e
    ceb = contract_eb_fast if fast else contract_eb
    cout = c2(H2, cin, norb, nelec, nmode, boson_states)
    cout += ceb(Heb, cin, norb, nelec, nmode, boson_states)
    cout += contract_bb(Hbb, cin, norb, nelec, nmode, boson_states)
    return cout


def check_contract_paths(H1, H2, Heb, Hbb, norb, nelec, nmode, boson_states,
                         seed=0):
    """Сверяет быстрый путь с эталонной питоновской реализацией."""
    rng = np.random.default_rng(seed)
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    v = rng.standard_normal(cishape)
    H2mod = absorb_h1e(H1, H2, norb, nelec, 0.5)
    a = contract_all(H1, H2mod, Heb, Hbb, v, norb, nelec, nmode,
                     boson_states, fast=False)
    b = contract_all(H1, H2mod, Heb, Hbb, v, norb, nelec, nmode,
                     boson_states, fast=True)
    return float(np.max(np.abs(a - b))), float(np.max(np.abs(a)))


def _slices_for(imode, nmode, fock_id, idxab=None):
    slices = [slice(None, None, None)] * (2 + nmode)
    slices[2 + imode] = fock_id
    if idxab is not None:
        ia, ib = idxab
        if ia is not None:
            slices[0] = ia
        if ib is not None:
            slices[1] = ib
    return tuple(slices)


def _slices_for_cre(imode, nmode, fock_id, idxab=None):
    return _slices_for(imode, nmode, fock_id + 1, idxab)


def contract_bb(Hbb, fcivec, norb, nelec, nmode, boson_states):
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)
    fcinew = np.zeros(cishape)
    for imode in range(nmode):
        nboson = boson_states[imode]
        for i in range(1, nboson + 1):
            s = _slices_for(imode, nmode, i)
            fcinew[s] += ci0[s] * (Hbb[imode, imode] * i)
    return fcinew.reshape(fcivec.shape)


def contract_eb(Heb, fcivec, norb, nelec, nmode, boson_states):
    """H_eb = sum_imode sum_pq Heb[imode,p,q] E_pq (b^+ + b).

    E_pq применяется один раз ко всему вектору, затем смешиваются соседние
    фоковские слои: электронный и бозонный сомножители коммутируют.
    """
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)
    fcinew = np.zeros(cishape, dtype=ci0.dtype)

    for imode in range(nmode):
        nboson = boson_states[imode]
        boson_cre = np.sqrt(np.arange(1, nboson + 1))

        Ec = np.zeros_like(ci0)
        for str0, tab in enumerate(link_indexa):
            for a, i, str1, sign in tab:
                Ec[str1] += sign * Heb[imode, a, i] * ci0[str0]
        for str0, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                Ec[:, str1] += sign * Heb[imode, a, i] * ci0[:, str0]

        for ip in range(nboson):
            s_low = _slices_for(imode, nmode, ip)
            s_high = _slices_for_cre(imode, nmode, ip)
            fcinew[s_high] += boson_cre[ip] * Ec[s_low]
            fcinew[s_low] += boson_cre[ip] * Ec[s_high]

    return fcinew.reshape(fcivec.shape)


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


def make_hdiag(h1e, eri, Hbb, norb, nelec, nmode, boson_states,
               floor=0.0):
    r"""Диагональ гамильтониана для предобусловливателя Дэвидсона.

    ВАЖНО: в исходном fci.py стояло max(i*omega, 0.5). При omega ~ 0.5
    (электронный масштаб) это почти точное значение и вреда не приносит,
    но при omega = 0.0064 (колебательный масштаб LiH) сдвиг фотонного
    сектора завышается в ~78 раз, предобусловливатель перестаёт работать
    и Дэвидсон сходится на порядки медленнее. Здесь по умолчанию берётся
    точное i*omega; floor > 0 возвращает прежнее поведение.
    """
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
            s = _slices_for(imode, nmode, i)
            hdiag[s] += max(i * Hbb[imode, imode], floor)
    return hdiag.ravel()


def qed_fci_kernel(H1, H2, norb, nelec, nmode, boson_states, Heb, Hbb,
                   ecore=0.0, tol=1e-10, lindep=1e-14, max_cycle=200,
                   max_space=30, seed=1, init_ci=None, hdiag_floor=0.0,
                   fast=True):
    if Hbb.ndim == 1:
        Hbb = np.diag(Hbb)
    np.random.seed(seed)
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)

    if init_ci is None or init_ci.shape != cishape:
        ci0 = np.zeros(cishape)
        ci0.__setitem__((0, 0) + (0,) * nmode, 1.0)
        ci0 += (np.random.random(cishape) - 0.5) * 1e-4
    else:
        ci0 = init_ci

    H2mod = absorb_h1e(H1, H2, norb, nelec, 0.5)

    def hop(c):
        return contract_all(H1, H2mod, Heb, Hbb, c, norb, nelec,
                            nmode, boson_states, fast=fast).reshape(-1)

    hdiag = make_hdiag(H1, H2, Hbb, norb, nelec, nmode, boson_states,
                       floor=hdiag_floor)
    precond = lib.make_diag_precond(hdiag, level_shift=1e-3)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond, tol=tol,
                        max_cycle=max_cycle, max_space=max_space,
                        verbose=0, nroots=1, lindep=lindep)
    return float(e) + ecore, c.reshape(cishape)


# ════════════════════════════════════════════════════════════════
#  Замораживание остова
# ════════════════════════════════════════════════════════════════
def freeze_core(h1, eri, ncore, ecore):
    r"""Стандартное преобразование замороженного остова.

        h1_act[p,q] = h1[p,q] + sum_{i in core} [ 2(pq|ii) - (pi|iq) ]
        ecore      += 2 sum_i h1[i,i] + sum_ij [ 2(ii|jj) - (ij|ji) ]

    Оно точно для ЛЮБОГО гамильтониана вида «одноэлектронный +
    двухэлектронный», поэтому применять его надо ПОСЛЕ того, как DSE-член
    свёрнут в h1 и eri: остовные и кор-валентные вклады дипольной
    самоэнергии тогда учитываются автоматически. Тензор D (х) D обладает
    нужной симметрией ((pq|rs) = (qp|rs) = (rs|pq)), так что формула
    применима без оговорок.

    Возвращает (h1_act, eri_act, ecore_new, act, core).
    """
    norb = h1.shape[0]
    if ncore <= 0:
        return h1, eri, ecore, slice(0, norb), slice(0, 0)

    core = slice(0, ncore)
    act = slice(ncore, norb)

    jcore = np.einsum('pqii->pq', eri[:, :, core, core])
    kcore = np.einsum('piiq->pq', eri[:, core, core, :])
    heff = h1 + 2.0 * jcore - kcore

    e_core = (2.0 * np.trace(h1[core, core])
              + 2.0 * np.einsum('iijj->', eri[core, core, core, core])
              - np.einsum('ijji->', eri[core, core, core, core]))

    return (np.ascontiguousarray(heff[act, act]),
            np.ascontiguousarray(eri[act, act, act, act]),
            ecore + e_core, act, core)


# ════════════════════════════════════════════════════════════════
#  Сборка гамильтониана
# ════════════════════════════════════════════════════════════════
def build_geometry_data(mol, omega, mode_vec, scf_conv_tol=1e-12,
                        scf_cycles=200, ncore=0, coherent_state=True,
                        z_fixed=None):
    """Один RHF и один ao2mo на геометрию; все полевые точки используют
    ОДНИ И ТЕ ЖЕ беспольевые МО. FCI инвариантен к вращению орбиталей,
    поэтому это законно, а E(F) выходит глаже (меньше FD-шума)."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = scf_conv_tol
    mf.max_cycle = scf_cycles
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('беспольевой RHF не сошёлся')

    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelectron
    mode_vec = np.asarray(mode_vec, dtype=float)

    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_full = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)

    dip_ao = mol.intor('int1e_r', comp=3)
    dip_mo = np.einsum('xpq,pi,qj->xij', dip_ao, mf.mo_coeff, mf.mo_coeff)
    D = np.einsum('x,xpq->pq', mode_vec, dip_mo)

    mu_nuc = np.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
    mu_nuc_lam = float(np.dot(mode_vec, mu_nuc))

    H1_full = h1e - mu_nuc_lam * D + 0.5 * (D @ D)
    H2_full = eri_full + np.einsum('pq,rs->pqrs', D, D)
    ecore_base = mol.energy_nuc() + 0.5 * mu_nuc_lam ** 2

    # ── когерентно-состоянческое преобразование ────────────────
    # U(z) = exp[z (b^+ - b)],  U^+ b U = b + z. В обозначениях кода
    # оператор связи есть Lambda = sum_pq D_pq E_pq - mu_nuc_lam, и
    # преобразование добавляет
    #     omega*z*(b^+ + b) - 2 z sqrt(w/2) Lambda + omega*z^2 .
    # При z = <Lambda>/sqrt(2w) три члена с DSE складываются в
    #     0.5*Lambda^2 - <Lambda>*Lambda + 0.5*<Lambda>^2 = 0.5*(Lambda-<Lambda>)^2,
    # то есть DSE становится членом с ФЛУКТУАЦИЕЙ диполя — как в QED-HF.
    #
    # Зачем: в исходном кадре основное состояние полярной молекулы обязано
    # построить смещение z = lambda*|mu|/sqrt(2*omega), и фотонный базис
    # должен вместить пуассоновское распределение со средним |z|^2. Для LiH
    # при lambda=0.1, omega=0.0064 это z ~ 2.05 и ~15 состояний Фока.
    # После преобразования смещение снимается и хватает 2-3 состояний.
    #
    # z берётся из хартри-фоковского диполя ОДИН РАЗ на геометрию и
    # одинаков для всех точек поля: так преобразование остаётся
    # фиксированным унитарным и не вносит в производные лишней
    # зависимости от поля. Энергия от выбора z не зависит вовсе (при
    # сошедшемся числе фотонов) — это встроенная проверка.
    z_cs = 0.0
    if coherent_state and omega > 0:
        if z_fixed is not None:
            z_cs = float(z_fixed)
        else:
            exp_lambda = float(np.einsum('p,pp->', mf.mo_occ, D)) - mu_nuc_lam
            z_cs = exp_lambda / np.sqrt(2.0 * omega)
        w2 = np.sqrt(omega / 2.0)
        H1_full = H1_full - 2.0 * z_cs * w2 * D
        ecore_base += omega * z_cs ** 2 + 2.0 * z_cs * w2 * mu_nuc_lam

    # ── замораживание остова (DSE уже внутри H1_full/H2_full) ──
    nelec_act = nelec - 2 * ncore
    if ncore > 0 and nelec_act < 2:
        raise ValueError(f'--frozen {ncore}: в активном пространстве '
                         f'осталось {nelec_act} электронов; для {nelec} '
                         f'электронов максимум {(nelec - 2) // 2}')
    H1_base, H2, ecore_base, act, core = freeze_core(
        H1_full, H2_full, ncore, ecore_base)
    norb_act = H1_base.shape[0]

    # Дипольные матрицы для внешнего поля: оператор одноэлектронный,
    # поэтому вклад остова — просто константа 2*sum_{i in core} dip_ii.
    dip_act = np.ascontiguousarray(dip_mo[:, act, act])
    dip_core = 2.0 * np.einsum('xii->x', dip_mo[:, core, core])

    # Билинейный член. Полный оператор связи:
    #     -sqrt(w/2) * [ sum_pq D_pq E_pq - mu_nuc_lam ]
    # В активном пространстве sum_pq D_pq E_pq = sum_act + 2*sum_core D_ii,
    # то есть добавляется константа. Константу вносим через
    # (const/nelec_act) * I, поскольку sum_pq delta_pq E_pq = N_el.
    D_core = 2.0 * np.trace(D[core, core]) if ncore > 0 else 0.0
    const = D_core - mu_nuc_lam
    Heb = np.zeros((1, norb_act, norb_act))
    Heb[0] = (-np.sqrt(omega / 2) * D[act, act]
              - np.sqrt(omega / 2) * (const / nelec_act) * np.eye(norb_act)
              + (omega * z_cs / nelec_act) * np.eye(norb_act))
    Hbb = np.array([[omega]])

    return dict(norb=norb_act, nelec=nelec_act, norb_full=norb, ncore=ncore,
                H1_base=H1_base, H2=H2, Heb=Heb, Hbb=Hbb,
                ecore_base=ecore_base, dip_mo=dip_act, dip_core=dip_core,
                mu_nuc=mu_nuc, mf=mf, z_cs=z_cs, D_full=D,
                mu_nuc_lam=mu_nuc_lam, omega=omega)


def expval_lambda(geom, civec, D_full, ncore):
    r"""<Lambda> = <sum_pq D_pq E_pq> - mu_nuc_lam по коррелированной функции.

    Электронный оператор действует внутри каждого фотонного слоя, поэтому
    одночастичная матрица плотности складывается послойно. Замороженный
    остов даёт постоянный вклад 2*sum_{i in core} D_ii.
    """
    norb, nelec = geom['norb'], geom['nelec']
    cishape = civec.shape
    na, nb = cishape[0], cishape[1]
    nph = int(np.prod(cishape[2:])) if len(cishape) > 2 else 1
    flat = civec.reshape(na, nb, nph)

    dm1 = np.zeros((norb, norb))
    for p in range(nph):
        layer = np.ascontiguousarray(flat[:, :, p])
        dm1 += direct_spin1.make_rdm1(layer, norb, nelec)

    act = slice(ncore, ncore + norb)
    exp_act = float(np.einsum('pq,pq->', D_full[act, act], dm1))
    exp_core = 2.0 * np.trace(D_full[:ncore, :ncore]) if ncore > 0 else 0.0
    return exp_act + exp_core - geom['mu_nuc_lam']


def _field_terms(geom, field_vec):
    """H' = -mu.F: электронная часть +F.<r>, ядерная -F.mu_nuc в ecore.
    При замороженном остове его вклад в электронную часть — константа
    +F.(2 sum_{i in core} dip_ii), она тоже идёт в ecore."""
    F = np.asarray(field_vec, dtype=float)
    H1 = geom['H1_base'] + np.einsum('x,xij->ij', F, geom['dip_mo'])
    ecore = (geom['ecore_base'] - float(np.dot(F, geom['mu_nuc']))
             + float(np.dot(F, geom['dip_core'])))
    return H1, ecore


def qed_fci_energy(geom, field_vec, nboson, fci_tol=1e-10, init_ci=None,
                   max_cycle=300, max_space=30, hdiag_floor=0.0, fast=True):
    H1, ecore = _field_terms(geom, field_vec)
    return qed_fci_kernel(H1, geom['H2'], geom['norb'], geom['nelec'],
                          nmode=1, boson_states=[nboson],
                          Heb=geom['Heb'], Hbb=geom['Hbb'],
                          ecore=ecore, tol=fci_tol, init_ci=init_ci,
                          max_cycle=max_cycle, max_space=max_space,
                          hdiag_floor=hdiag_floor, fast=fast)


def dse_only_energy(geom, field_vec, fci_tol=1e-10):
    """FCI с DSE, но без билинейной связи — предел omega -> 0.
    При Heb = 0 бозонный сектор отщепляется, остаётся электронный FCI
    с DSE-модифицированными интегралами."""
    H1, ecore = _field_terms(geom, field_vec)
    e, _ = direct_spin1.kernel(H1, geom['H2'], geom['norb'], geom['nelec'],
                               ecore=ecore, tol=fci_tol, verbose=0)
    return float(np.ravel(e)[0])


# ════════════════════════════════════════════════════════════════
#  Аргументы
# ════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--mol', default='H2', choices=sorted(MOLECULES))
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--omega', type=float, default=0.4687)
    p.add_argument('--mode-dir', type=float, nargs=3, default=[0., 0., 1.],
                   metavar=('X', 'Y', 'Z'),
                   help='направление поляризации моды, нормируется автоматически')
    p.add_argument('--full-tensor', action='store_true',
                   help='полный тензор даже при осевой моде (проверка симметрии)')

    p.add_argument('--basis-file', default='sadlej.gbs',
                   help='путь к .gbs ИЛИ имя встроенного базиса PySCF')
    p.add_argument('--basis-label', default='sadlej')

    p.add_argument('--rmin', type=float, default=0.5)
    p.add_argument('--rmax', type=float, default=4.0)
    p.add_argument('--rstep', type=float, default=0.1)

    p.add_argument('--ff-step', type=float, default=0.004)
    p.add_argument('--fast', action='store_true',
                   help='a_xx только по ±h (7 точек вместо 9)')

    p.add_argument('--nboson', type=int, default=6,
                   help='максимальное число фотонов (состояний Фока nboson+1). '
                        'Замер для H2 при lambda=0.1: шаги сходимости '
                        '2e-6 (nb=3), 7e-8 (4), 2e-9 (6), 2e-12 (8) — '
                        'шестёрки хватает с запасом')
    p.add_argument('--boson-check', default='ends',
                   choices=['none', 'ends', 'all'],
                   help='где проверять сходимость по числу фотонов')
    p.add_argument('--boson-check-list', type=int, nargs='+', default=None,
                   help='значения nboson для проверки сходимости. По '
                        'умолчанию подбираются относительно --nboson '
                        '(половина и на два меньше), чтобы зазор считался '
                        'между БЛИЗКИМИ значениями и говорил о сходимости '
                        'именно рабочего nboson, а не о недостаточности '
                        'заведомо малого')
    p.add_argument('--no-dse-only', action='store_true',
                   help='не считать предел omega->0 (FCI+DSE без фотонной связи)')

    p.add_argument('--slow-contract', action='store_true',
                   help='эталонная питоновская свёртка вместо C-кода PySCF '
                        '(на порядки медленнее, только для отладки)')
    p.add_argument('--no-frame-check', action='store_true',
                   help='не сверять энергию с КС-преобразованием и без него '
                        'на первой точке (энергия от выбора кадра не зависит)')
    p.add_argument('--no-contract-check', action='store_true',
                   help='не сверять быстрый путь с эталонным на первой точке')
    p.add_argument('--hdiag-floor', type=float, default=0.0,
                   help='нижняя отсечка сдвига фотонного сектора в '
                        'предобусловливателе; 0 = точное i*omega (по '
                        'умолчанию). Значение 0.5 воспроизводит поведение '
                        'исходного fci.py — не используйте при малых omega')
    p.add_argument('--fci-tol', type=float, default=1e-11)
    p.add_argument('--fci-cycles', type=int, default=300)
    p.add_argument('--max-space', type=int, default=30)
    p.add_argument('--scf-conv', type=float, default=1e-12)
    p.add_argument('--scf-cycles', type=int, default=200)

    p.add_argument('--max-fail', type=int, default=3)
    p.add_argument('--z-damp', type=float, default=0.0, metavar='F',
                   help='демпфирование самосогласования z: '
                        'z_new = (1-F)*z_target + F*z_old. Помогает, если '
                        'итерации колеблются вокруг решения')
    p.add_argument('--z-tol', type=float, default=1e-3, metavar='EPS',
                   help='порог сходимости z по остаточному числу фотонов '
                        '|z_new^2 - z_old^2|')
    p.add_argument('--z-iter', type=int, default=3, metavar='N',
                   help='итераций самосогласования когерентного сдвига по '
                        'коррелированной плотности (0 = взять z из RHF, как '
                        'делает QED-HF в OpenMS). Нужно там, где среднее поле '
                        'плохо описывает диполь — у растянутого LiH требуется '
                        '3-4 итерации, в равновесии хватает одной')
    p.add_argument('--no-coherent-state', action='store_true',
                   help='не делать когерентно-состоянческое преобразование. '
                        'Для полярных молекул при малых omega это резко '
                        'увеличивает нужное число фотонов: смещение равно '
                        'lambda*|mu|/sqrt(2*omega), и базис должен вместить '
                        'пуассоновское распределение со средним |z|^2')
    p.add_argument('--frozen', type=int, default=0, metavar='N',
                   help='заморозить N низших МО (для LiH: --frozen 1, это '
                        '1s^2 лития). Проверено на LiH/Sadlej при lambda=0: '
                        'отклонение alpha от полного FCI -0.15..-0.31%% и '
                        'почти постоянно по R, поэтому в разности '
                        'alpha(lambda)-alpha(0) сокращается')
    p.add_argument('--no-center', action='store_true',
                   help='не центрировать геометрию')
    p.add_argument('--verify', action='store_true',
                   help='при lam=0 сверить QED-FCI с обычным FCI из PySCF')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--outdir', default='.')
    return p.parse_args(argv)


def log(msg=''):
    print(msg, flush=True)


# ════════════════════════════════════════════════════════════════
#  Симметрия (как в qedcc_scan.py)
# ════════════════════════════════════════════════════════════════
def normalise_mode_dir(mode_dir):
    v = np.asarray(mode_dir, float).ravel()
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError('--mode-dir не может быть нулевым вектором')
    if abs(n - 1.0) > 1e-10:
        log(f'  --mode-dir нормирован: {v.tolist()} -> '
            f'{np.round(v / n, 8).tolist()}')
    return v / n


def choose_scheme(mode_dir, force_full=False, fast=False):
    u = np.asarray(mode_dir, float)
    if (not force_full) and abs(abs(u[2]) - 1.0) < 1e-10:
        return 'axial', [2, 0], [], {2: True, 0: not fast}
    on = [abs(u[i]) > 1e-10 for i in range(3)]
    offdiag = [(i, j) for i in range(3) for j in range(i + 1, 3)
               if on[i] and on[j]]
    return 'general', [0, 1, 2], offdiag, {0: True, 1: True, 2: True}


def field_tags(axes, offdiag, rich):
    tags = ['0']
    for i in axes:
        a = AXIS[i]
        tags += [f'+{a}', f'-{a}']
        if rich.get(i, False):
            tags += [f'+2{a}', f'-2{a}']
    for (i, j) in offdiag:
        ai, aj = AXIS[i], AXIS[j]
        tags += [f'+{ai}+{aj}', f'+{ai}-{aj}', f'-{ai}+{aj}', f'-{ai}-{aj}']
    return tags


# ════════════════════════════════════════════════════════════════
#  Молекула и базис
# ════════════════════════════════════════════════════════════════
def load_basis(basis_spec, elements):
    path = basis_spec
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            basis_spec)
    if os.path.exists(path):
        basis = {}
        for el in elements:
            try:
                basis[el] = parse_gaussian.load(path, el)
            except Exception as exc:
                raise RuntimeError(
                    f'в {os.path.basename(path)} нет блока для {el!r} '
                    f'({exc})') from exc
        return basis
    try:
        basis = {el: gto.basis.load(basis_spec, el) for el in elements}
    except Exception as exc:
        raise FileNotFoundError(
            f'{basis_spec!r}: не найден ни как файл, ни как встроенный '
            f'базис PySCF ({exc})') from exc
    log(f'  базис {basis_spec!r} из встроенной библиотеки PySCF')
    return basis


def make_mol(cfg, basis, R, center=True):
    mol = gto.Mole()
    mol.atom = cfg['template'].format(R=R)
    mol.basis = basis
    mol.charge = cfg['charge']
    mol.spin = cfg['spin']
    mol.unit = 'Angstrom'
    mol.verbose = 0
    mol.build()
    if center:
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        zcen = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_geom_(coords - zcen, unit='Bohr')
    return mol


# ════════════════════════════════════════════════════════════════
#  Конечные разности (идентично qedcc_scan.py)
# ════════════════════════════════════════════════════════════════
def derive_props(E, axes, offdiag, rich, h):
    r"""E(F) = E0 - mu.F - (1/2) F.alpha.F - ..."""
    mu = np.zeros(3)
    alpha = np.zeros((3, 3))
    err = np.full(3, np.nan)
    E0 = E['0']

    for i in axes:
        a = AXIS[i]
        Ep, Em = E[f'+{a}'], E[f'-{a}']
        a_h = -(Ep - 2 * E0 + Em) / h**2
        if rich.get(i, False):
            E2p, E2m = E[f'+2{a}'], E[f'-2{a}']
            mu[i] = -(-E2p + 8 * Ep - 8 * Em + E2m) / (12 * h)
            a_2h = -(E2p - 2 * E0 + E2m) / (2 * h)**2
            alpha[i, i] = (4 * a_h - a_2h) / 3
            err[i] = abs(a_h - a_2h)
        else:
            mu[i] = -(Ep - Em) / (2 * h)
            alpha[i, i] = a_h

    for (i, j) in offdiag:
        ai, aj = AXIS[i], AXIS[j]
        alpha[i, j] = alpha[j, i] = -(
            E[f'+{ai}+{aj}'] - E[f'+{ai}-{aj}']
            - E[f'-{ai}+{aj}'] + E[f'-{ai}-{aj}']) / (4 * h**2)

    if 1 not in axes:
        alpha[1, 1] = alpha[0, 0]
        err[1] = err[0]
    return mu, alpha, err


def _field_list(axes, offdiag, rich, h):
    e = np.eye(3)
    out = [('0', np.zeros(3))]
    for i in axes:
        a = AXIS[i]
        out += [(f'+{a}', h * e[i]), (f'-{a}', -h * e[i])]
        if rich.get(i, False):
            out += [(f'+2{a}', 2 * h * e[i]), (f'-2{a}', -2 * h * e[i])]
    for (i, j) in offdiag:
        ai, aj = AXIS[i], AXIS[j]
        out += [(f'+{ai}+{aj}',  h * (e[i] + e[j])),
                (f'+{ai}-{aj}',  h * (e[i] - e[j])),
                (f'-{ai}+{aj}', -h * (e[i] - e[j])),
                (f'-{ai}-{aj}', -h * (e[i] + e[j]))]
    return out


def ff_props(geom, args, axes, offdiag, rich):
    """QED-FCI с тёплым стартом CI-вектора между точками поля."""
    h = args.ff_step
    en = {}
    ci_prev = None
    for tag, f in _field_list(axes, offdiag, rich, h):
        t0 = time.time()
        val, civec = qed_fci_energy(geom, f, args.nboson,
                                    fci_tol=args.fci_tol, init_ci=ci_prev,
                                    max_cycle=args.fci_cycles,
                                    max_space=args.max_space,
                                    hdiag_floor=args.hdiag_floor,
                                  fast=not args.slow_contract)
        ci_prev = civec
        en[tag] = val
        log(f'      [{tag:>5s}]  E = {val: .12f}   ({time.time() - t0:5.1f} s)')
    return derive_props(en, axes, offdiag, rich, h), en


def dse_only_props(geom, args, axes, offdiag, rich):
    """Тот же набор точек поля в пределе omega -> 0."""
    h = args.ff_step
    en = {}
    for tag, f in _field_list(axes, offdiag, rich, h):
        en[tag] = dse_only_energy(geom, f, fci_tol=args.fci_tol)
    return derive_props(en, axes, offdiag, rich, h)


def _check_list(args):
    if args.boson_check_list:
        return sorted(set(list(args.boson_check_list) + [args.nboson]))
    nb = args.nboson
    cand = {max(2, nb // 2), max(2, nb - 2), nb}
    return sorted(cand)


def boson_convergence(geom, args):
    nbs = _check_list(args)
    energies = {}
    prev = None
    for nb in nbs:
        t0 = time.time()
        # тёплый старт: вектор с меньшим nboson дополняем нулями
        init = None
        if prev is not None:
            shape = fboson_ci_shape(geom['norb'], geom['nelec'], 1, [nb])
            init = np.zeros(shape)
            sl = tuple(slice(0, min(a, b))
                       for a, b in zip(shape, prev.shape))
            init[sl] = prev[sl]
        e, civec = qed_fci_energy(geom, np.zeros(3), nb, fci_tol=args.fci_tol,
                                  init_ci=init, max_cycle=args.fci_cycles,
                                  max_space=args.max_space,
                                  hdiag_floor=args.hdiag_floor,
                                  fast=not args.slow_contract)
        prev = civec
        energies[nb] = e
        log(f'      nboson={nb}: E = {e: .12f}   ({time.time() - t0:5.1f} s)')
    gap = abs(energies[nbs[-1]] - energies[nbs[-2]]) if len(nbs) > 1 else np.nan
    log(f'      зазор между двумя наибольшими nboson: {gap:.2e} Ha')
    if np.isfinite(gap) and gap > 1e-6:
        log(f'      !! поднимите --nboson выше {args.nboson}')
    elif np.isfinite(gap) and gap > 1e-8:
        log('      (зазор мал, на alpha практически не влияет — '
            'в разностях сокращается)')
    return gap


def refine_z(mol, args, mode_vec, geom, n_iter):
    """Самосогласовывает когерентный сдвиг по КОРРЕЛИРОВАННОЙ плотности.

    Начальное z берётся из хартри-фоковского диполя. Для растянутого LiH
    RHF держит ионную конфигурацию Li+H- и завышает диполь (при R=5.5 A
    даёт z = 6.04 против самосогласованного -0.07), поэтому сдвиг
    оказывается не там, где нужно, и фотонный базис приходится раздувать.

    Условие стационарности: z = <Lambda>/sqrt(2*omega) с точной волновой
    функцией. Энергия от выбора z не зависит вовсе (при сошедшемся числе
    фотонов), поэтому итерации улучшают только сходимость по фотонам.

    Критерий остановки задан по ОСТАТОЧНОМУ числу фотонов, а не по самому
    dz: значение имеет |z|^2, поэтому вблизи нуля допуск можно ослабить.
    Демпфирование гасит колебания вокруг решения.
    """
    if n_iter <= 0 or args.no_coherent_state or args.omega <= 0:
        return geom

    ci_prev = None
    for it in range(n_iter):
        _, civec = qed_fci_energy(geom, np.zeros(3), args.nboson,
                                  fci_tol=args.fci_tol, init_ci=ci_prev,
                                  max_cycle=args.fci_cycles,
                                  max_space=args.max_space,
                                  hdiag_floor=args.hdiag_floor,
                                  fast=not args.slow_contract)
        ci_prev = civec
        exp_lam = expval_lambda(geom, civec, geom['D_full'], geom['ncore'])
        z_target = exp_lam / np.sqrt(2.0 * geom['omega'])
        z_old = geom['z_cs']
        z_new = (1.0 - args.z_damp) * z_target + args.z_damp * z_old
        dz = z_new - z_old
        dn = abs(z_new ** 2 - z_old ** 2)
        log(f'    самосогласование z: итерация {it + 1}:  '
            f'{z_old:.4f} -> {z_new:.4f}  (dz = {dz:+.4f}, '
            f'd<n> = {dn:.2e})')

        geom = build_geometry_data(mol, args.omega, mode_vec,
                                   scf_conv_tol=args.scf_conv,
                                   scf_cycles=args.scf_cycles,
                                   ncore=args.frozen,
                                   coherent_state=True, z_fixed=z_new)
        ci_prev = None            # форма гамильтониана та же, но кадр сменился

        # остаточное смещение важно через <n> = z^2; вблизи нуля допуск мягче
        if dn < args.z_tol:
            log(f'    z сошёлся за {it + 1} итераций '
                f'(остаточное d<n> = {dn:.2e} < {args.z_tol:.0e})')
            break
    else:
        if n_iter > 1:
            log(f'    !! z не сошёлся за {n_iter} итераций; '
                f'увеличьте --z-iter или включите --z-damp')
    return geom


def print_props(mu, alpha, err, dse=None):
    log(f'      mu (a.u.)        x = {mu[0]: .6f}   y = {mu[1]: .6f}   '
        f'z = {mu[2]: .6f}   |mu| = {np.linalg.norm(mu): .6f}')
    if dse is None:
        log('      alpha (a.u.)          QED-FCI')
        for i in range(3):
            row = ' '.join(f'{alpha[i, j]: 10.4f}' for j in range(3))
            log(f'          {AXIS[i]}       [{row} ]')
        log(f'          iso      {np.trace(alpha) / 3: 10.4f}')
    else:
        d = alpha - dse[1]
        log('      alpha (a.u.)        QED-FCI                  '
            'предел omega->0            фотонный вклад')
        for i in range(3):
            r1 = ' '.join(f'{alpha[i, j]: 9.4f}' for j in range(3))
            r2 = ' '.join(f'{dse[1][i, j]: 9.4f}' for j in range(3))
            r3 = ' '.join(f'{d[i, j]: 9.4f}' for j in range(3))
            log(f'          {AXIS[i]}     [{r1} ]   [{r2} ]   [{r3} ]')
        i1, i2 = np.trace(alpha) / 3, np.trace(dse[1]) / 3
        log(f'          iso    {i1: 9.4f}{"":25s}{i2: 9.4f}'
            f'{"":25s}{i1 - i2: 9.4f}')
    log(f'      err (Ричардсон):  xx {err[0]:.2e}   yy {err[1]:.2e}   '
        f'zz {err[2]:.2e}')


def point_status(alpha, err, gap, ff_step=None):
    flags = []
    # ошибка дискретизации после Ричардсона ~ (err/16); если она заметна
    # на фоне alpha, шаг поля великоват — для LiH в хвосте это главное
    if np.any(np.diag(alpha) <= 0):
        flags.append('NEG_ALPHA')
    # err — расхождение шаблонов h и 2h; после Ричардсона остаётся ~err/16.
    # 2% — уже мусор, 0.5% — сигнал уменьшить шаг поля.
    for i, name in enumerate(('XX', 'YY', 'ZZ')):
        if not np.isfinite(err[i]) or abs(alpha[i, i]) < 1e-8:
            continue
        rel = err[i] / abs(alpha[i, i])
        if rel > 2e-2:
            flags.append(f'FD_NOISE_{name}')
        elif rel > 5e-3:
            flags.append(f'STEP_{name}')
    # Порог по энергии, а не по alpha: ошибка усечения по фотонам почти
    # одинакова во всех точках поля и в конечных разностях сокращается.
    # Замер при lambda=0.1, R=0.74: зазор 6.8e-8 Ha по энергии отвечает
    # расхождению alpha_zz всего 1e-4 а.е. (6.4718 против 6.4719), поэтому
    # 1e-6 здесь уже с большим запасом.
    if np.isfinite(gap) and gap > 1e-6:
        flags.append(f'BOSON_GAP={gap:.1e}')
    return 'OK' if not flags else '+'.join(flags)


# ════════════════════════════════════════════════════════════════
#  Проверка при lambda = 0
# ════════════════════════════════════════════════════════════════
def verify_against_plain_fci(cfg, basis, args):
    mol = make_mol(cfg, basis, cfg['Re'], center=not args.no_center)
    field = np.array([0., 0., args.ff_step])
    geom = build_geometry_data(mol, args.omega, np.zeros(3),
                               scf_conv_tol=args.scf_conv,
                               scf_cycles=args.scf_cycles,
                               ncore=args.frozen,
                               coherent_state=not args.no_coherent_state)
    e_ref = dse_only_energy(geom, field, fci_tol=args.fci_tol)
    e_qed, _ = qed_fci_energy(geom, field, args.nboson, fci_tol=args.fci_tol,
                              max_cycle=args.fci_cycles,
                              max_space=args.max_space,
                              hdiag_floor=args.hdiag_floor,
                                  fast=not args.slow_contract)
    log(f'  сверка при lambda=0:  FCI = {e_ref: .12f}   '
        f'QED-FCI = {e_qed: .12f}   разница = {abs(e_qed - e_ref):.2e} Ha')
    if abs(e_qed - e_ref) > 1e-8:
        log('  !! расхождение больше 1e-8 Ha — разберитесь до продакшн-скана')

    # при замороженном остове дополнительно сверяем с CASCI из PySCF:
    # это независимая проверка преобразования остова
    if args.frozen > 0:
        from pyscf import mcscf
        # ВАЖНО: CASCI должен идти от ТЕХ ЖЕ беспольевых орбиталей, что и
        # наш расчёт. Если дать ему орбитали, релаксированные в поле, то
        # заморожена окажется другая орбиталь, и сравнение будет некорректным
        # (для LiH при h=0.004 расхождение выходит ~2e-7 Ha — это разница
        # определений активного пространства, а не ошибка преобразования).
        with mol.with_common_orig((0., 0., 0.)):
            ao_r = mol.intor_symmetric('int1e_r', comp=3)
        h_field = (mol.intor_symmetric('int1e_kin')
                   + mol.intor_symmetric('int1e_nuc')
                   + np.einsum('x,xij->ij', field, ao_r))
        mf2 = geom['mf']
        saved_hcore = mf2.get_hcore
        mf2.get_hcore = lambda *a, **k: h_field
        try:
            ncas = mf2.mo_coeff.shape[1] - args.frozen
            nelecas = mol.nelectron - 2 * args.frozen
            mc = mcscf.CASCI(mf2, ncas, nelecas)
            mc.verbose = 0
            mc.fcisolver.conv_tol = args.fci_tol
            mc.kernel(mf2.mo_coeff)      # беспольевые орбитали
            e_cas = float(mc.e_tot)
        finally:
            mf2.get_hcore = saved_hcore
        log(f'  сверка замороженного остова (те же орбитали):  '
            f'CASCI = {e_cas: .12f}   наш FCI = {e_ref: .12f}   '
            f'разница = {abs(e_cas - e_ref):.2e} Ha')
        if abs(e_cas - e_ref) > 1e-8:
            log('  !! преобразование остова расходится с CASCI')
    return abs(e_qed - e_ref)


# ════════════════════════════════════════════════════════════════
#  Метаданные
# ════════════════════════════════════════════════════════════════
def meta_header(args, cfg, scheme, offdiag, tags, comment='#'):
    import platform
    try:
        import pyscf
        psc = pyscf.__version__
    except Exception:
        psc = 'unknown'
    items = [
        ('molecule', args.mol), ('geometry', cfg['template']),
        ('basis', args.basis_label), ('basis_source', args.basis_file),
        ('lambda', f'{args.lam:.6f}'), ('omega', f'{args.omega:.6f}'),
        ('mode_dir', [round(float(x), 8) for x in args.mode_dir]),
        ('scheme', scheme),
        ('offdiag', [AXIS[i] + AXIS[j] for i, j in offdiag] or 'none'),
        ('method', 'QED-FCI'), ('nboson_max', args.nboson),
        ('n_fock_states', args.nboson + 1),
        ('ff_step_h', args.ff_step), ('n_field_points', len(tags)),
        ('fci_tol', args.fci_tol), ('fci_max_cycle', args.fci_cycles),
        ('scf_conv_tol', args.scf_conv),
        ('center_geometry', not args.no_center),
        ('frozen_core_orbitals', args.frozen),
        ('coherent_state', not args.no_coherent_state),
        ('z_selfconsistent_iter', args.z_iter),
        ('R_range_Ang', f'{args.rmin}:{args.rstep}:{args.rmax}'),
        ('R_e_ref_Ang', cfg['Re']),
        ('pyscf', psc), ('python', platform.python_version()),
        ('host', platform.node()),
        ('date', time.strftime('%Y-%m-%d %H:%M:%S')),
    ]
    return '\n'.join(f'{comment} {k:16s} = {v}' for k, v in items)


def done_points(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith('#'):
                continue
            p = line.split()
            if p:
                try:
                    out.add(round(float(p[0]), 4))
                except ValueError:
                    pass
    return out


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════
def main(argv=None):
    args = parse_args(argv)
    cfg = MOLECULES[args.mol]

    mode_dir = normalise_mode_dir(args.mode_dir)
    args.mode_dir = list(mode_dir)
    scheme, axes, offdiag, rich = choose_scheme(
        mode_dir, force_full=args.full_tensor, fast=args.fast)
    tags = field_tags(axes, offdiag, rich)
    mode_vec = args.lam * mode_dir

    os.makedirs(args.outdir, exist_ok=True)
    tag = (f'{args.mol}_lam{args.lam:.3f}_om{args.omega:.4f}_'
           f'{args.basis_label}_fci_nb{args.nboson}'
           + (f'_fc{args.frozen}' if args.frozen else ''))
    if scheme != 'axial':
        tag += '_pol' + ''.join(f'{x:+.2f}' for x in mode_dir).replace('.', 'p')
    datfile = os.path.join(args.outdir, f'{tag}_data.txt')
    enfile = os.path.join(args.outdir, f'{tag}_energies.txt')

    basis = load_basis(args.basis_file, cfg['elements'])

    log('\n' + '#' * 92)
    log(f'#  {args.mol}   QED-FCI   lambda = {args.lam}   '
        f'omega = {args.omega}   {args.basis_label}')
    log(f'#  поляризация {np.round(mode_dir, 6).tolist()}   схема: {scheme}'
        + (f'   недиагональные: {[AXIS[i] + AXIS[j] for i, j in offdiag]}'
           if offdiag else ''))
    if args.frozen:
        log(f'#  заморожено {args.frozen} низших МО (остов)')
    log(f'#  фотонов до {args.nboson} ({args.nboson + 1} состояний Фока);   '
        f'h = {args.ff_step};   {len(tags)} FCI-расчётов на точку'
        + ('' if args.no_dse_only else ' + столько же для предела omega->0'))
    log(f'#  R = {args.rmin}-{args.rmax} A, шаг {args.rstep}')
    log('#' * 92)

    if args.verify:
        if abs(args.lam) > 1e-12:
            log('  --verify имеет смысл только при lambda = 0, пропускаю')
        else:
            verify_against_plain_fci(cfg, basis, args)

    R_vals = np.arange(args.rmin, args.rmax + args.rstep / 2, args.rstep)
    skip = done_points(datfile) if args.resume else set()
    if skip:
        log(f'  --resume: пропускаю {len(skip)} готовых точек')

    fmode = 'a' if (args.resume and os.path.exists(datfile)) else 'w'
    fh = open(datfile, fmode)
    fe = open(enfile, fmode)
    if fmode == 'w':
        head = meta_header(args, cfg, scheme, offdiag, tags)
        fh.write(head + '\n')
        fh.write('# columns          = ' + ','.join(COLUMNS) + '\n')
        fh.write('# ' + '  '.join(f'{c:>13s}' for c in COLUMNS) + '\n')
        fe.write(head + '\n')
        fe.write('# columns          = R,'
                 + ','.join(f'fci[{t}]' for t in tags) + '\n')
        fh.flush()
        fe.flush()

    t_start, n_fail, n_ok = time.time(), 0, 0

    for k, R in enumerate(R_vals):
        if round(float(R), 4) in skip:
            continue
        log(f'\n  R = {R:.2f} A   [{k + 1}/{len(R_vals)}, '
            f'прошло {time.time() - t_start:7.0f} s]')
        try:
            mol = make_mol(cfg, basis, R, center=not args.no_center)
            t0 = time.time()
            geom = build_geometry_data(mol, args.omega, mode_vec,
                                       scf_conv_tol=args.scf_conv,
                                       scf_cycles=args.scf_cycles,
                                       ncore=args.frozen,
                                       coherent_state=not args.no_coherent_state)
            log(f'    RHF + ao2mo: {time.time() - t0:.1f} s   '
                f'(norb = {geom["norb"]})')
            geom = refine_z(mol, args, mode_vec, geom, args.z_iter)

            if abs(geom['z_cs']) > 1e-12:
                nph = geom['z_cs'] ** 2
                log(f'    когерентный сдвиг z = {geom["z_cs"]:.4f}   '
                    f'<n_phot> = {nph:.2f} снято преобразованием '
                    f'(иначе понадобилось бы ~{int(nph + 5 * np.sqrt(nph)) + 2} '
                    f'состояний Фока)')

            # при lambda = 0 связи нет, фотонные секторы точно вырождены,
            # и проверять сходимость по их числу нечего
            if k == 0 and not args.no_frame_check and args.lam != 0:
                geom_nocs = build_geometry_data(
                    mol, args.omega, mode_vec, scf_conv_tol=args.scf_conv,
                    scf_cycles=args.scf_cycles, ncore=args.frozen,
                    coherent_state=args.no_coherent_state)
                e_a, _ = qed_fci_energy(geom, np.zeros(3), args.nboson,
                                        fci_tol=args.fci_tol,
                                        hdiag_floor=args.hdiag_floor)
                e_b, _ = qed_fci_energy(geom_nocs, np.zeros(3), args.nboson,
                                        fci_tol=args.fci_tol,
                                        hdiag_floor=args.hdiag_floor)
                log(f'    инвариантность к выбору кадра: '
                    f'с КС {e_a:.12f}   без КС {e_b:.12f}   '
                    f'разница {abs(e_a - e_b):.2e} Ha')
                if abs(e_a - e_b) > 1e-7:
                    log('      (расхождение = недосходимость по числу фотонов '
                        'в том кадре, где смещение больше; это ожидаемо)')

            if k == 0 and not args.no_contract_check:
                dev, scale = check_contract_paths(
                    geom['H1_base'], geom['H2'], geom['Heb'], geom['Hbb'],
                    geom['norb'], geom['nelec'], 1, [args.nboson])
                log(f'    сверка быстрой свёртки с эталонной: '
                    f'max|разность| = {dev:.2e}  (масштаб {scale:.2e})')
                if dev > 1e-9 * max(scale, 1.0):
                    raise RuntimeError('быстрая свёртка расходится с эталонной')

            do_check = (abs(args.lam) > 1e-12
                        and (args.boson_check == 'all'
                             or (args.boson_check == 'ends'
                                 and k in (0, len(R_vals) - 1))))
            gap = np.nan
            if do_check:
                log('    проверка сходимости по числу фотонов (F=0):')
                gap = boson_convergence(geom, args)

            (mu, alpha, err), en = ff_props(geom, args, axes, offdiag, rich)
            dse = None if args.no_dse_only else dse_only_props(
                geom, args, axes, offdiag, rich)

            status = point_status(alpha, err, gap, args.ff_step)
            print_props(mu, alpha, err, dse)
            log(f'      статус: [{status}]')

            if dse is None:
                dse_vals = [np.nan] * 5
            else:
                dse_vals = [dse[0][2], dse[1][0, 0], dse[1][1, 1],
                            dse[1][2, 2], float(np.trace(dse[1]) / 3)]

            vals = ([R, float(np.linalg.norm(mu))] + list(mu)
                    + [alpha[0, 0], alpha[1, 1], alpha[2, 2], alpha[0, 2],
                       float(np.trace(alpha) / 3)]
                    + list(err) + dse_vals + [gap])
            fh.write('  '.join(f'{v:13.6f}' if abs(v) < 1e5 else f'{v:13.4e}'
                               for v in vals) + f'  {status}\n')
            fe.write(f'{R:12.6f} ' + ' '.join(
                f'{en.get(t, float("nan")):24.16f}' for t in tags) + '\n')
            fh.flush()
            fe.flush()

            n_ok += 1
            n_fail = 0
            if status != 'OK':
                log('      !! точка ненадёжна')

        except Exception as exc:
            log(f'      СБОЙ: {exc}')
            n_fail += 1
            fh.write(f'{R:13.6f}  FAILED  # {exc}\n')
            fh.flush()
            if n_fail >= args.max_fail:
                log(f'\n  {n_fail} срыва подряд — скан остановлен на '
                    f'R = {R:.2f} A')
                break

    fh.close()
    fe.close()
    log(f'\nГотово: {n_ok} точек за {time.time() - t_start:.0f} s')
    log(f'  данные:  {datfile}')
    log(f'  энергии: {enfile}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
