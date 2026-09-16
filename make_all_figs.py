#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единый скрипт отрисовки ВСЕХ фигур статьи (PRA style, все в EPS).

Рисует все 10 фигур, включённых в main.tex:
  fig1_h2_alpha_R          -- альфа(R) для H2, обе компоненты
  fig2_h2_relative_R       -- относительное изменение альфа_zz(R), H2
  fig3_decomposition       -- разложение на DSE и фотонный обмен
  fig_cc_vs_fci            -- E(lambda): QED-HF/CCSD/FCI, H2, Sadlej
  fig4_lih_props_R         -- дипольный момент и альфа(R) для LiH
  fig5_lih_relative_R      -- относительное изменение альфа_zz(R), LiH
  fig8_isotropic_R         -- изотропная альфа(R), H2 и LiH
  fig9_h2_orientation      -- параллельная vs перпендикулярная мода
  fig6_he_states           -- относительное изменение альфа для He
  fig7_he_differential     -- дифференциальная альфа перехода He

Легенды теперь полностью непрозрачные (framealpha=1.0, до этого было
0.85) -- если планируете двигать легенды руками поверх кривых, это
не мешает, линии под непрозрачным фоном просто не видны, что и нужно.

Все фигуры пишутся сразу в трёх форматах: .eps (основной, векторный,
без проблем с прозрачностью в PostScript), .pdf (для быстрой проверки
через pdflatex), .png (для просмотра глазами).

Данные читаются через qedlib.py (fig1-fig9, fig6-fig7 -- ожидает
data/qedfci_frozen_cs+z_with_opt/ и data/qedfci_atom/ рядом со
скриптом, либо work/... -- см. qedlib.ROOT) и напрямую из
cc_lambda_scan/ + fci_lambda_scan/ рядом со скриптом (fig_cc_vs_fci).
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qedlib as Q

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs2')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'axes.linewidth': 0.7, 'lines.linewidth': 1.2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
    'legend.frameon': True, 'legend.framealpha': 1.0,
    'legend.facecolor': 'white', 'legend.edgecolor': 'none',
    'legend.handlelength': 2.0,
    'legend.labelspacing': 0.25, 'legend.borderpad': 0.2,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
COL1 = 3.375
LBL_FS = 7.5   # single font size for every panel label and legend title


def lighten(color, factor=0.55):
    """Смешивает цвет с белым в пропорции factor (заменяет alpha=factor
    поверх белого фона). EPS/PostScript не поддерживает настоящую
    прозрачность мазка, поэтому вместо alpha= используем заранее
    посчитанный светлый оттенок того же цвета."""
    r, g, b = mcolors.to_rgb(color)
    return (1 - factor + factor * r, 1 - factor + factor * g,
            1 - factor + factor * b)

LAMS = [0.0, 0.01, 0.025, 0.05, 0.1]
STY = {0.0:   ('k',       '-',  r'$\lambda=0$'),
       0.01:  ('#0072B2', '--', r'$\lambda=0.01$'),
       0.025: ('#009E73', '-.', r'$\lambda=0.025$'),
       0.05:  ('#D55E00', ':',  r'$\lambda=0.05$'),
       0.1:   ('#CC79A7', (0, (3, 1, 1, 1, 1, 1)), r'$\lambda=0.10$')}
ANG = r'$R$ ($\mathrm{\AA}$)'


def save(fig, name):
    for ext in ('eps', 'pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'),
                    dpi=(600 if ext == 'png' else None))
    plt.close(fig)
    print('  wrote', name, '(eps, pdf, png)')


def vline(ax, x):
    ax.axvline(x, color='0.6', lw=0.6, ls=(0, (1, 2)), zorder=0)


def clip(d, lo, hi):
    m = (d['R'] >= lo - 1e-9) & (d['R'] <= hi + 1e-9)
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# ════════════════════════════════════════════════════════════
# FIG 1. H2 absolute polarizabilities vs R
# ════════════════════════════════════════════════════════════
def fig1():
    s = {l: clip(d, 0.5, 4.0) for l, d in Q.mol_set('h2', 'el').items()}
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 4.2), sharex=True)
    for l in LAMS:
        c, ls, lab = STY[l]
        ax[0].plot(s[l]['R'], s[l]['a_xx'], color=c, ls=ls, label=lab)
        ax[1].plot(s[l]['R'], s[l]['a_zz'], color=c, ls=ls)
    ax[0].set_ylabel(r'$\alpha_{xx}$ (a.u.)')
    ax[1].set_ylabel(r'$\alpha_{zz}$ (a.u.)')
    ax[1].set_xlabel(ANG)
    ax[1].set_xlim(0.5, 4.0)
    for k, a in enumerate(ax):
        vline(a, Q.RE['H2'])
        a.xaxis.set_minor_locator(AutoMinorLocator(2))
        a.yaxis.set_minor_locator(AutoMinorLocator(2))
        a.text(0.03, 0.90, '(%s)' % 'ab'[k], transform=a.transAxes, va='top',
               fontsize=LBL_FS)
    leg = ax[0].legend(loc='lower right',
                       title=r'H$_2$, $\omega_{\rm c}=0.4687$ a.u.')
    leg.get_title().set_fontsize(LBL_FS)
    fig.subplots_adjust(hspace=0.06)
    save(fig, 'fig1_h2_alpha_R')

# ════════════════════════════════════════════════════════════
# FIG 2. H2 relative change vs R, two frequencies
# ════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 4.2), sharex=True)
    for k, (reg, om) in enumerate((('el', 0.4687), ('vib', 0.0201))):
        s = Q.mol_set('h2', reg)
        ref = s[0.0]
        for l in LAMS[1:]:
            c, ls, lab = STY[l]
            R, dz = Q.common_grid(ref, s[l], 'a_zz')
            m = (R >= 0.5) & (R <= 4.0)
            ax[k].plot(R[m], dz[m], color=c, ls=ls, label=lab)
        ax[k].axhline(0, color='0.75', lw=0.5, zorder=0)
        vline(ax[k], Q.RE['H2'])
        ax[k].set_ylabel(r'$\Delta\alpha_{zz}/\alpha_{zz}^{(0)}$ (%)')
        ax[k].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[k].yaxis.set_minor_locator(AutoMinorLocator(2))
        # подпись панели -- внутри рамки, по центру
        ax[k].text(0.5, 0.5, r'(%s) $\omega_{\rm c}=%.4f$ a.u.'
                   % ('ab'[k], om), transform=ax[k].transAxes,
                   ha='center', va='center', fontsize=LBL_FS)
    ax[1].set_xlabel(ANG)
    ax[1].set_xlim(0.5, 4.0)
    # легенда -- обычным боксом внутри нижней панели, в левом нижнем углу
    handles, labels_ = ax[0].get_legend_handles_labels()
    leg = ax[1].legend(handles, labels_, loc='lower left', ncol=1,
                       fontsize=LBL_FS)
    fig.subplots_adjust(hspace=0.06)
    save(fig, 'fig2_h2_relative_R')

# ════════════════════════════════════════════════════════════
# FIG 3. Decomposition vs lambda at equilibrium (central result)
# ════════════════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 4.4), sharex=True)
    panels = [('h2', 'H$_2$', [('el', 0.4687), ('vib', 0.0201)]),
              ('lih', 'LiH', [('el', 0.1208), ('vib', 0.0064)])]
    for k, (mol, name, regs) in enumerate(panels):
        Re = Q.RE['H2' if mol == 'h2' else 'LiH']
        s0 = Q.mol_set(mol, regs[0][0])
        a0 = Q.at(s0[0.0], Re, 'a_zz')
        lams = sorted(l for l in s0 if l > 0)

        dse = [100 * (Q.at(s0[l], Re, 'dse_a_zz') / a0 - 1) for l in lams]
        ax[k].plot([0] + lams, [0] + dse, 'o-', ms=3.5, color='k',
                   label=r'DSE only ($\omega_{\rm c}\!\to\!0$)')
        mk = ['s', '^']
        cl = ['#D55E00', '#0072B2']
        for j, (reg, om) in enumerate(regs):
            s = Q.mol_set(mol, reg)
            full = [100 * (Q.at(s[l], Re, 'a_zz') / a0 - 1) for l in lams]
            ax[k].plot([0] + lams, [0] + full, mk[j] + '--', ms=3.5,
                       color=cl[j],
                       label=r'QED-FCI, $\omega_{\rm c}=%.4f$' % om)
        ax[k].axhline(0, color='0.75', lw=0.5, zorder=0)
        ax[k].set_ylabel(r'$\Delta\alpha_{zz}/\alpha_{zz}^{(0)}$ (%)')
        ax[k].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[k].yaxis.set_minor_locator(AutoMinorLocator(2))
        leg = ax[k].legend(loc='lower left',
                           title=f'({"ab"[k]}) {name}, $R=R_e$')
        leg.get_title().set_fontsize(LBL_FS)
    ax[1].set_xlabel(r'$\lambda$ (a.u.)')
    ax[1].set_xlim(-0.003, 0.103)
    fig.subplots_adjust(hspace=0.06)
    save(fig, 'fig3_decomposition')


# ════════════════════════════════════════════════════════════
# FIG 4. LiH dipole and polarizabilities vs R
# ════════════════════════════════════════════════════════════
def fig4():
    s = {l: clip(d, 0.8, 6.0) for l, d in Q.mol_set('lih', 'vib').items()}
    fig, ax = plt.subplots(3, 1, figsize=(COL1, 5.6), sharex=True)
    for l in LAMS:
        c, ls, lab = STY[l]
        ax[0].plot(s[l]['R'], -s[l]['mu_z'], color=c, ls=ls, label=lab)
        ax[1].plot(s[l]['R'], s[l]['a_xx'], color=c, ls=ls)
        ax[2].plot(s[l]['R'], s[l]['a_zz'], color=c, ls=ls)
    ax[0].set_ylabel(r'$|\mu_z|$ (a.u.)')
    ax[1].set_ylabel(r'$\alpha_{xx}$ (a.u.)')
    ax[2].set_ylabel(r'$\alpha_{zz}$ (a.u.)')
    ax[2].set_xlabel(ANG)
    ax[2].set_xlim(0.8, 6.0)
    for k, a in enumerate(ax):
        vline(a, Q.RE['LiH'])
        a.xaxis.set_minor_locator(AutoMinorLocator(2))
        a.yaxis.set_minor_locator(AutoMinorLocator(2))
        if k:
            a.text(0.03, 0.90, '(%s)' % 'abc'[k], transform=a.transAxes,
                   va='top', fontsize=LBL_FS)
    leg = ax[0].legend(loc='lower left',
                       title=r'(a) LiH, $\omega_{\rm c}=0.0064$ a.u.')
    leg.get_title().set_fontsize(LBL_FS)
    fig.subplots_adjust(hspace=0.06)
    save(fig, 'fig4_lih_props_R')


# ════════════════════════════════════════════════════════════
# FIG 5. LiH relative change vs R
# ════════════════════════════════════════════════════════════
def fig5():
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 4.2), sharex=True)
    for k, (reg, om) in enumerate((('el', 0.1208), ('vib', 0.0064))):
        s = Q.mol_set('lih', reg)
        ref = s[0.0]
        for l in LAMS[1:]:
            c, ls, lab = STY[l]
            R, dz = Q.common_grid(ref, s[l], 'a_zz')
            m = (R >= 0.8) & (R <= 3.0)
            ax[k].plot(R[m], dz[m], color=c, ls=ls, label=lab)
        ax[k].axhline(0, color='0.75', lw=0.5, zorder=0)
        vline(ax[k], Q.RE['LiH'])
        ax[k].set_ylabel(r'$\Delta\alpha_{zz}/\alpha_{zz}^{(0)}$ (%)')
        ax[k].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[k].yaxis.set_minor_locator(AutoMinorLocator(2))
        if k:
            ax[k].text(0.03, 0.06, '(%s) $\\omega_{\\rm c}=%.4f$ a.u.'
                       % ('ab'[k], om), transform=ax[k].transAxes,
                       fontsize=LBL_FS)
    ax[1].set_xlabel(ANG)
    ax[1].set_xlim(0.8, 3.0)
    leg = ax[0].legend(loc='center left', ncol=2, columnspacing=1.0,
                       title=r'(a) $\omega_{\rm c}=0.1208$ a.u.')
    leg.get_title().set_fontsize(LBL_FS)
    fig.subplots_adjust(hspace=0.06)
    save(fig, 'fig5_lih_relative_R')


# ════════════════════════════════════════════════════════════
# FIG 6. Helium: state-resolved polarizabilities vs lambda
# ════════════════════════════════════════════════════════════
def fig6():
    A = Q.atom_set()
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 4.4), sharex=True)
    names = {'g': r'$1^1S_0$', 't': r'$2^3S_1$', 's': r'$2^1S_0$'}
    cols = {'g': '#0072B2', 't': '#D55E00', 's': '#009E73'}
    for k, om in enumerate((0.1000, 0.0293)):
        for st in ('g', 't', 's'):
            d = A.get((st, om))
            if d is None:
                continue
            # выше lambda = 0.05 состояния 2^3S и 2^1S сильно смешаны с
            # фотонными репликами 2P (перекрытие с беспольевым состоянием
            # падает ниже 0.9), поэтому такие точки не показываем
            m = d['lambda'] <= 0.0501
            l = d['lambda'][m]
            a0z, a0x = d['a_zz'][0], d['a_xx'][0]
            ax[k].plot(l, 100 * (d['a_zz'][m] / a0z - 1), 'o-', ms=3.5,
                       color=cols[st], label=names[st])
            ax[k].plot(l, 100 * (d['a_xx'][m] / a0x - 1), 's--', ms=3,
                       color=lighten(cols[st]))
        ax[k].axhline(0, color='0.75', lw=0.5, zorder=0)
        ax[k].set_ylabel(r'$\Delta\alpha_{ii}/\alpha_{ii}^{(0)}$ (%)')
        ax[k].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[k].yaxis.set_minor_locator(AutoMinorLocator(2))
    leg = ax[0].legend(loc='upper left',
                       title=r'(a) He, $\omega_{\rm c}=0.1000$ a.u.')
    leg.get_title().set_fontsize(LBL_FS)
    h = [Line2D([], [], color='0.35', ls='-', marker='o', ms=3.5, label=r'$zz$'),
         Line2D([], [], color=lighten('0.35'), ls='--', marker='s', ms=3,
                label=r'$xx$')]
    leg2 = ax[1].legend(handles=h, loc='upper left',
                        title=r'(b) He, $\omega_{\rm c}=0.0293$ a.u.')
    leg2.get_title().set_fontsize(LBL_FS)
    ax[1].set_xlabel(r'$\lambda$ (a.u.)')
    ax[1].set_xlim(-0.002, 0.053)
    fig.subplots_adjust(hspace=0.06)
    save(fig, 'fig6_he_states')


# ════════════════════════════════════════════════════════════
# FIG 7. Helium: differential polarizability and BBR shift
# ════════════════════════════════════════════════════════════
def fig7():
    A = Q.atom_set()
    AU_HZ = 6.579683920502e15
    E2 = (831.9 / 5.14220675e11) ** 2       # <E^2> at 300 K, a.u.
    fig, ax = plt.subplots(figsize=(COL1, 2.9))
    cl = {0.1000: '#0072B2', 0.0293: '#D55E00'}
    for om in (0.1000, 0.0293):
        t, s = A.get(('t', om)), A.get(('s', om))
        n = min(len(t['lambda']), len(s['lambda']))
        lam = t['lambda'][:n]
        d_iso = s['a_iso'][:n] - t['a_iso'][:n]
        d_zz = s['a_zz'][:n] - t['a_zz'][:n]
        ax.plot(lam, d_iso, 'o-', ms=3.5, color=cl[om],
                label=r'$\overline{\Delta\alpha}$, $\omega_{\rm c}=%.4f$' % om)
        ax.plot(lam, d_zz, 's--', ms=3, color=lighten(cl[om]),
                label=r'$\Delta\alpha_{zz}$, $\omega_{\rm c}=%.4f$' % om)
    ax.set_xlabel(r'$\lambda$ (a.u.)')
    ax.set_ylabel(r'$\Delta\alpha=\alpha(2^1S_0)-\alpha(2^3S_1)$ (a.u.)')
    ax.set_xlim(-0.003, 0.053)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc='upper left', ncol=1, fontsize=LBL_FS)

    sec = ax.secondary_yaxis(
        'right',
        functions=(lambda a: -0.5 * E2 * a * AU_HZ,
                   lambda h: -2.0 * h / (E2 * AU_HZ)))
    sec.set_ylabel(r'$\delta\nu_{\rm BBR}$ at 300 K (Hz)')
    sec.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.16, right=0.80, bottom=0.16, top=0.96)
    save(fig, 'fig7_he_differential')

# ════════════════════════════════════════════════════════════
# FIG 8. Isotropic polarizability vs R, both molecules
# ════════════════════════════════════════════════════════════
def fig8():
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 4.2))
    mols = [('h2', 'el', 0.4687, r'H$_2$', 0.5, 4.0),
            ('lih', 'vib', 0.0064, r'LiH', 0.8, 6.0)]
    for k, (mol, reg, om, name, lo, hi) in enumerate(mols):
        s = {l: clip(d, lo, hi) for l, d in Q.mol_set(mol, reg).items()}
        for l in LAMS:
            c, ls, lab = STY[l]
            ax[k].plot(s[l]['R'], s[l]['a_iso'], color=c, ls=ls, label=lab)
        vline(ax[k], Q.RE['H2' if mol == 'h2' else 'LiH'])
        ax[k].set_ylabel(r'$\bar{\alpha}$ (a.u.)')
        ax[k].set_xlim(lo, hi)
        ax[k].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[k].yaxis.set_minor_locator(AutoMinorLocator(2))
        if k == 0:
            leg = ax[k].legend(loc='center', bbox_to_anchor=(0.5, 0.4),
                               ncol=1, columnspacing=1.0,
                               title=rf'({"ab"[k]}) {name}, $\omega_{{\rm c}}={om:.4f}$')
        else:
            leg = ax[k].legend(loc='upper left',
                               ncol=1, columnspacing=1.0,
                               title=rf'({"ab"[k]}) {name}, $\omega_{{\rm c}}={om:.4f}$')
        leg.get_title().set_fontsize(LBL_FS)
    ax[1].set_xlabel(ANG)
    fig.subplots_adjust(hspace=0.28)
    save(fig, 'fig8_isotropic_R')


# ════════════════════════════════════════════════════════════
# FIG 9. H2, cavity polarization parallel vs perpendicular
# to the molecular axis
# ════════════════════════════════════════════════════════════
def fig9():
    fig, ax = plt.subplots(2, 2, figsize=(2 * COL1 + 0.3, 4.4), sharex=True)
    mols = [('h2', Q.RE['H2'], [('el', 0.4687), ('vib', 0.0201)]),
            ('lih', Q.RE['LiH'], [('el', 0.1208), ('vib', 0.0064)])]
    for col, (mol, Re, regs) in enumerate(mols):
        for k, (reg, om) in enumerate(regs):
            par = Q.mol_set(mol, reg, perp=False)
            perp = Q.mol_set(mol, reg, perp=True)
            lams = sorted(l for l in par if l > 0)
            a0_par = Q.at(par[0.0], Re, 'a_iso')
            a0_perp = Q.at(perp[0.0], Re, 'a_iso')
            d_par = [100 * (Q.at(par[l], Re, 'a_iso') / a0_par - 1) for l in lams]
            d_perp = [100 * (Q.at(perp[l], Re, 'a_iso') / a0_perp - 1)
                     for l in lams]
            a = ax[k, col]
            a.plot([0] + lams, [0] + d_par, 'o-', ms=3.5, color='#D55E00',
                  label=r'$\boldsymbol{\lambda}\parallel$ bond axis')
            a.plot([0] + lams, [0] + d_perp, 's--', ms=3.5, color='#0072B2',
                  label=r'$\boldsymbol{\lambda}\perp$ bond axis')
            a.axhline(0, color='0.75', lw=0.5, zorder=0)
            a.xaxis.set_minor_locator(AutoMinorLocator(2))
            a.yaxis.set_minor_locator(AutoMinorLocator(2))
            panel = '(%s)' % 'abcd'[2 * col + k]
            leg = a.legend(loc='lower left',
                           title=rf'{panel} $\omega_{{\rm c}}={om:.4f}$ a.u.')
            leg.get_title().set_fontsize(LBL_FS)
        ax[0, col].set_title(r'H$_2$' if mol == 'h2' else 'LiH', fontsize=9)
        ax[1, col].set_xlabel(r'$\lambda$ (a.u.)')
    ax[0, 0].set_ylabel(r'$\Delta\bar{\alpha}/\bar{\alpha}^{(0)}$ (%)')
    ax[1, 0].set_ylabel(r'$\Delta\bar{\alpha}/\bar{\alpha}^{(0)}$ (%)')
    ax[0, 0].set_xlim(-0.003, 0.103)
    fig.subplots_adjust(hspace=0.06, wspace=0.28)
    save(fig, 'fig9_h2_orientation')



# ════════════════════════════════════════════════════════════
# FIG cc_vs_fci. E(lambda) comparison: QED-HF, QED-CCSD-U22-S2,
# QED-FCI for H2 at R=Re (Sadlej basis, omega=0.4687)
# ════════════════════════════════════════════════════════════
def load(path):
    cols, lam = None, None
    for line in open(path):
        if line.startswith('#'):
            if 'columns' in line:
                cols = [c.strip() for c in line.split('=', 1)[1].split(',')]
            elif 'lambda' in line and '=' in line:
                lam = float(line.split('=', 1)[1])
            continue
        vals = [float(x) for x in line.split()]
        row = dict(zip(cols, vals))
    return lam, row


def collect():
    # данные для этой фигуры лежат рядом со скриптом (не в figs2/, куда
    # сохраняются картинки) -- отдельная переменная, чтобы не путать с OUT
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for f_cc in sorted(glob.glob(os.path.join(script_dir, 'cc_lambda_scan',
                                              '*_energies.txt'))):
        lam, row_cc = load(f_cc)
        cand = glob.glob(os.path.join(
            script_dir, 'fci_lambda_scan',
            f'H2_lam{lam:.3f}_om0.4687_sadlej_fci_nb8_energies.txt'))
        _, row_fci = load(cand[0])
        rows.append(dict(lam=lam, e_hf=row_cc['hf[0]'], e_cc=row_cc['cc[0]'],
                         e_fci=row_fci['fci[0]']))
    rows.sort(key=lambda r: r['lam'])
    return rows


def fig_cc_vs_fci():
    rows = collect()
    lam = np.array([r['lam'] for r in rows])
    e_hf = np.array([r['e_hf'] for r in rows])
    e_cc = np.array([r['e_cc'] for r in rows])
    e_fci = np.array([r['e_fci'] for r in rows])

    d_hf = 1000 * (e_hf - e_hf[0])
    d_cc = 1000 * (e_cc - e_cc[0])
    d_fci = 1000 * (e_fci - e_fci[0])

    fig, ax = plt.subplots(2, 1, figsize=(COL1, 4.4), sharex=True)

    ax[0].plot(lam, d_hf, 'D-', ms=4, color='0.4', label='QED-HF')
    ax[0].plot(lam, d_cc, 's--', ms=4.5, color='#D55E00',
              label='QED-CCSD-U22-S2')
    ax[0].plot(lam, d_fci, 'o-', ms=4.5, color='#0072B2', label='QED-FCI')
    ax[0].set_ylabel(r'$E(\lambda)-E(0)$ (mHa)')
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
    leg = ax[0].legend(loc='upper left', title=r'(a) H$_2$, $R=R_e$')
    leg.get_title().set_fontsize(LBL_FS)

    ratio = d_cc[1:] / d_fci[1:]
    ax[1].plot(lam[1:], ratio, 'o-', ms=4.5, color='#D55E00')
    ax[1].axhline(1.0, color='0.6', lw=0.8, ls=(0, (4, 2)))
    ax[1].set_ylabel(r'$[E_{\rm CCSD}(\lambda)-E_{\rm CCSD}(0)]\,/\,'
                     r'[E_{\rm FCI}(\lambda)-E_{\rm FCI}(0)]$')
    ax[1].set_xlabel(r'$\lambda$ (a.u.)')
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
    ax[1].set_xlim(-0.005, 0.155)
    ax[1].text(0.05, 0.50, r'(b) QED-CCSD overestimates the cavity shift'
              '\nby a factor of $\\approx 1.26$ at all $\\lambda$',
              transform=ax[1].transAxes, va='top', fontsize=LBL_FS)

    fig.subplots_adjust(hspace=0.06)
    for ext in ('eps', 'pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_cc_vs_fci.{ext}'),
                    dpi=(600 if ext == 'png' else None))
    plt.close(fig)

    print(f'{"lambda":>8s} {"E_HF":>15s} {"E_CCSD":>15s} {"E_FCI":>15s} '
          f'{"dE_CCSD(mHa)":>13s} {"dE_FCI(mHa)":>12s} {"ratio":>7s}')
    for i, r in enumerate(rows):
        rat = d_cc[i] / d_fci[i] if i else float('nan')
        print(f'{r["lam"]:8.3f} {r["e_hf"]:15.9f} {r["e_cc"]:15.9f} '
              f'{r["e_fci"]:15.9f} {d_cc[i]:13.4f} {d_fci[i]:12.4f} '
              f'{rat:7.3f}')




if __name__ == '__main__':
    print('Generating figures ...')
    fig1(); fig2(); fig3(); fig_cc_vs_fci(); fig4(); fig5(); fig8(); fig9()
    fig6(); fig7()
    print('done ->', OUT)
