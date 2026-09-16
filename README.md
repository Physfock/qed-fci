# QED-FCI: exact static electric response of a molecule in an optical cavity (python implementation and data used in the original manuscript)

`qedfci_scan_cs_sz.py` computes the dipole moment and the full static
polarizability tensor of a diatomic molecule coupled to a single quantized
cavity mode, by full configuration interaction (FCI) diagonalization of the
Pauli–Fierz Hamiltonian in the coherent-state frame. Properties are extracted
as finite-field derivatives of the total energy.

The implementation has been validated against an
independent code (the [eT program](https://etprogram.org)) to better than
1 nHa.

## Physics implemented

- **Pauli–Fierz Hamiltonian**, length gauge, single cavity mode, in the
  coherent-state frame. The coherent-state displacement `z` can be taken
  directly from the mean-field (RHF) dipole, or determined
  self-consistently from the *correlated* FCI density (`--z-iter`), which is
  necessary whenever the mean-field dipole is a poor estimate of the true
  one — e.g. a stretched, polar molecule such as LiH near dissociation.
- **Dipole self-energy (DSE)** is folded directly into the one- and
  two-electron integrals before diagonalization ($h_{pq} \mathrel{+}= \tfrac12(DD)_{pq}$,
  $(pq|rs) \mathrel{+}= D_{pq}D_{rs}$, with the *asymmetric* 0.5/1.0 split
  required by how PySCF's `absorb_h1e` folds one-electron terms into the
  two-electron tensor — a naive symmetric 0.5/0.5 split silently gives the
  wrong energy).
- **Bilinear (photon-exchange) term** is applied separately as a
  one-electron operator that moves amplitude between adjacent photon-number
  blocks, using PySCF's own `direct_spin1` machinery for the
  electronic contraction inside each block.
- **Frozen core** (`--frozen N`) folds the N lowest occupied orbitals into
  an effective one-electron operator plus a constant shift, exactly as in
  CASCI, applied *after* the DSE has already been added to the integrals —
  so it automatically captures the core contribution to the self-energy too.
- **Finite-field response**: $\mu_i = -\partial E/\partial F_i$,
  $\alpha_{ij} = -\partial^2 E/\partial F_i \partial F_j$, evaluated with a
  Richardson-extrapolated central-difference stencil (two step sizes, $h$
  and $2h$) to push the leading truncation error from $\mathcal O(h^2)$ to
  $\mathcal O(h^4)$. The number of field points needed depends on how the
  cavity mode is oriented relative to the molecular axis (7–9 points along
  the axis, up to 25 for a fully general off-axis mode).
- **Convergence checks built in**: photon-number convergence
  (`--boson-check`), coherent-state frame invariance of the energy
  (on by default, disable with `--no-frame-check`), agreement between the
  fast C-accelerated contraction and a reference Python implementation
  (`--no-contract-check`), and an optional direct comparison against plain
  PySCF FCI at $\lambda=0$ (`--verify`).

## Supported molecules from the box

`H2`, `LiH`, `HF`, `HeH+` (add new diatomics by extending the `MOLECULES`
dictionary at the top of the file).

## Requirements

- Python 3
- [PySCF](https://pyscf.org)
- NumPy

## Usage

```bash
# Single point, with a sanity check against plain FCI at lambda = 0
python qedfci_scan_cs_sz.py --mol H2 --lam 0.00 --omega 0.4687 \
    --rmin 0.74 --rmax 0.74 --verify

# Bond-length scan at finite coupling
python qedfci_scan_cs_sz.py --mol H2 --lam 0.10 --omega 0.4687 \
    --rmin 0.5 --rmax 4.0

# Polar molecule near dissociation: self-consistent coherent-state shift
# and a generous photon-number cutoff
python qedfci_scan_cs_sz.py --mol LiH --lam 0.10 --omega 0.0064 \
    --rmin 5.0 --rmax 5.0 --basis-file cc-pvdz --basis-label ccpvdz \
    --nboson 20 --z-iter 4 --boson-check all
```

### Key arguments

| Flag | Default | Meaning |
|---|---|---|
| `--mol` | `H2` | Molecule, one of `H2`, `LiH`, `HF`, `HeH+` |
| `--lam` | `0.0` | Coupling strength $\lambda$ (a.u.) |
| `--omega` | `0.4687` | Cavity mode frequency $\omega_{\rm c}$ (a.u.) |
| `--mode-dir X Y Z` | `0 0 1` | Cavity polarization direction (auto-normalized) |
| `--basis-file` | `sadlej.gbs` | Path to a `.gbs` file, or the name of a basis built into PySCF |
| `--rmin --rmax --rstep` | `0.5 4.0 0.1` | Bond-length scan range, Å |
| `--ff-step` | `0.004` | Finite-field step size $h$ |
| `--fast` | off | Use a 3-point stencil for the off-axis component (7 points instead of 9) |
| `--nboson` | `6` | Photon Fock-space cutoff $N_{\rm p}$ |
| `--z-iter` | `3` | Self-consistency iterations for the coherent-state shift, from the correlated density (`0` uses the RHF dipole instead) |
| `--frozen` | `0` | Number of lowest occupied orbitals to freeze |
| `--boson-check` | `ends` | Where to verify photon-number convergence: `none`, `ends`, or `all` points |
| `--verify` | off | At $\lambda=0$, cross-check against plain PySCF FCI |
| `--outdir` | `.` | Output directory |

Run `python qedfci_scan_cs_sz.py --help` for the complete list, including
solver tolerances and numerical-stability options.

## Output

For each run the script writes two tab-separated files into `--outdir`:

- `*_data.txt` — dipole moment, polarizability tensor components, DSE-only
  reference values, the Richardson-extrapolation error estimate, and the
  photon-number convergence gap, one row per bond length.
- `*_energies.txt` — the raw total energy at every finite-field point of
  the stencil, useful for independent post-processing or for comparison
  against other codes at the energy level.
