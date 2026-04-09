"""
Test batched per-atom parameter correction vs sequential single-molecule.

Success criteria (with tight SCF convergence x_atol=1e-10, f_atol=1e-10):
  1. E_batch[i] ≈ E_seq[i]   (energy match)
  2. F_batch[i] ≈ F_seq[i]   (force match)
  3. dE/dp_batch ≈ dE/dp_seq  (parameter gradient match)
"""

from __future__ import annotations

import pytest
import torch
import dxtb
import tomli as toml
from dxtb._src.param.base import Param
from tad_mctc.batch import pack
from pathlib import Path

DEVICE = torch.device("cpu")

ANG2BOHR = 1.8897261328856432

# SCF options: tight convergence
TIGHT_OPTS = {
    "verbosity": 0,
    "maxiter": 300,
    "x_atol": 1e-10,
    "f_atol": 1e-10,
}

# All element param keys and their per-atom shapes
ELEM_PARAMS = {
    "levels": 3, "slater": 3, "shpoly": 3, "kcn": 3,
    "gam": 1, "lgam": 3, "arad": 1, "rcov": 1,
    "gam3": 1, "zeff": 1, "arep": 1, "en": 1,
    "dkernel": 1, "qkernel": 1, "mprad": 1, "mpvcn": 1,
    "3rd_scale": 3, "qsh": 3, "predicted_energy": 3,
}

PAIR_PARAMS = [
    "theta_ss", "theta_pp", "theta_dd", "theta_sp", "theta_sd", "theta_pd",
    "zeta_ss", "zeta_pp", "zeta_dd", "zeta_sp", "zeta_sd", "zeta_pd",
]

_CACHED_TOML = None


def _load_toml():
    global _CACHED_TOML
    if _CACHED_TOML is None:
        toml_path = (
            Path(dxtb.__file__).parent / "_src" / "param" / "gfn2" / "gfn2-xtb.toml"
        )
        with open(toml_path, "rb") as fd:
            _CACHED_TOML = toml.load(fd)
    return _CACHED_TOML


def _make_full_params(natom, small=0.01, seed=42):
    """Create a complete per-atom correction dict with small random values."""
    torch.manual_seed(seed)
    params = {}
    for key, length in ELEM_PARAMS.items():
        shape = (natom,) if length == 1 else (natom, length)
        params[key] = (torch.randn(shape) * small).requires_grad_(True)
    for key in PAIR_PARAMS:
        params[key] = (torch.randn(natom, natom) * small).requires_grad_(True)
    return params


def _make_batched_params(per_mol_params):
    """Pack per-molecule param dicts into batched tensors (detached, new leaf)."""
    batched = {}
    for key in per_mol_params[0]:
        batched[key] = pack(
            [p[key].detach() for p in per_mol_params], value=0.0,
        ).requires_grad_(True)
    return batched


# Test molecules
MOLECULES = [
    ("H2O", torch.tensor([8, 1, 1], device=DEVICE),
     torch.tensor([
         [0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692],
     ], dtype=torch.float64, device=DEVICE) * ANG2BOHR),
    ("CH4", torch.tensor([6, 1, 1, 1, 1], device=DEVICE),
     torch.tensor([
         [0.0, 0.0, 0.0], [0.6276, 0.6276, 0.6276],
         [0.6276, -0.6276, -0.6276], [-0.6276, 0.6276, -0.6276],
         [-0.6276, -0.6276, 0.6276],
     ], dtype=torch.float64, device=DEVICE) * ANG2BOHR),
    ("LiH", torch.tensor([3, 1], device=DEVICE),
     torch.tensor([
         [0.0, 0.0, 0.0], [0.0, 0.0, 1.596],
     ], dtype=torch.float64, device=DEVICE) * ANG2BOHR),
]


def _run_sequential(mol_indices):
    """Run molecules one-by-one. Returns energies, forces, dE/d(en)."""
    energies, forces, grads = [], [], []
    for idx in mol_indices:
        name, numbers, positions = MOLECULES[idx]
        pos = positions.clone().requires_grad_(True)
        params = _make_full_params(len(numbers), seed=100 + idx)

        par = Param(**_load_toml())
        par._per_atom_params_dict = params
        calc = dxtb.Calculator(
            numbers, par, opts={**TIGHT_OPTS, "batch_mode": 0},
            dtype=torch.float64,
        )
        e = calc.energy(pos)
        energies.append(e.detach())

        (neg_f,) = torch.autograd.grad(e, pos, retain_graph=True)
        forces.append((-neg_f).detach())

        (g,) = torch.autograd.grad(e, params["en"], retain_graph=True)
        grads.append(g.detach())

    return energies, forces, grads


def _run_batched(mol_indices):
    """Run molecules as padded batch. Returns energies, forces, dE/d(en)."""
    mols = [MOLECULES[i] for i in mol_indices]
    per_mol_params = [
        _make_full_params(len(m[1]), seed=100 + i)
        for i, m in zip(mol_indices, mols)
    ]

    numbers_batch = pack([m[1] for m in mols], value=0)
    positions_batch = pack(
        [m[2].clone() for m in mols], value=0.0,
    ).requires_grad_(True)
    batched_params = _make_batched_params(per_mol_params)

    par = Param(**_load_toml())
    par._per_atom_params_dict = batched_params
    chrg = torch.zeros(len(mols), dtype=torch.float64)

    calc = dxtb.Calculator(
        numbers_batch, par,
        opts={**TIGHT_OPTS, "batch_mode": 1},
        dtype=torch.float64,
    )
    e = calc.energy(positions_batch, chrg=chrg)

    (neg_f,) = torch.autograd.grad(e.sum(), positions_batch, retain_graph=True)
    (g,) = torch.autograd.grad(e.sum(), batched_params["en"], retain_graph=True)

    natoms = [len(m[1]) for m in mols]
    energies = [e[i].detach() for i in range(len(mols))]
    forces = [(-neg_f[i, :natoms[i]]).detach() for i in range(len(mols))]
    grads = [g[i, :natoms[i]].detach() for i in range(len(mols))]

    return energies, forces, grads


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSameMoleculeBatch:
    """Batching identical molecules should match to machine precision."""

    def test_2x_h2o(self):
        e_seq, f_seq, g_seq = _run_sequential([0, 0])
        e_bat, f_bat, g_bat = _run_batched([0, 0])
        for i in range(2):
            assert torch.allclose(e_seq[i], e_bat[i], atol=1e-12), \
                f"E mismatch: {(e_seq[i] - e_bat[i]).abs().item():.2e}"
            assert torch.allclose(f_seq[i], f_bat[i], atol=1e-12), \
                f"F mismatch: {(f_seq[i] - f_bat[i]).abs().max().item():.2e}"
            assert torch.allclose(g_seq[i], g_bat[i], atol=1e-12), \
                f"G mismatch: {(g_seq[i] - g_bat[i]).abs().max().item():.2e}"

    def test_2x_ch4(self):
        e_seq, f_seq, g_seq = _run_sequential([1, 1])
        e_bat, f_bat, g_bat = _run_batched([1, 1])
        for i in range(2):
            assert torch.allclose(e_seq[i], e_bat[i], atol=1e-12)
            assert torch.allclose(f_seq[i], f_bat[i], atol=1e-12)
            assert torch.allclose(g_seq[i], g_bat[i], atol=1e-12)


class TestDifferentMoleculeBatch:
    """Batching different molecules — tolerance depends on SCF convergence."""

    @pytest.fixture(autouse=True)
    def _results(self):
        """Compute sequential and batched results once per class."""
        self.mol_indices = [0, 1, 2]  # H2O, CH4, LiH
        self.e_seq, self.f_seq, self.g_seq = _run_sequential(self.mol_indices)
        self.e_bat, self.f_bat, self.g_bat = _run_batched(self.mol_indices)

    def test_energy_match(self):
        for i, idx in enumerate(self.mol_indices):
            diff = (self.e_seq[i] - self.e_bat[i]).abs().item()
            assert diff < 1e-6, \
                f"{MOLECULES[idx][0]} E_diff={diff:.2e} > 1e-6"

    def test_force_match(self):
        for i, idx in enumerate(self.mol_indices):
            diff = (self.f_seq[i] - self.f_bat[i]).abs().max().item()
            assert diff < 1e-5, \
                f"{MOLECULES[idx][0]} F_diff={diff:.2e} > 1e-5"

    def test_param_grad_match(self):
        for i, idx in enumerate(self.mol_indices):
            diff = (self.g_seq[i] - self.g_bat[i]).abs().max().item()
            assert diff < 1e-6, \
                f"{MOLECULES[idx][0]} G_diff={diff:.2e} > 1e-6"


class TestDifferentMoleculeBatchPairwise:
    """Test various 2-molecule combinations."""

    @pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
    def test_energy(self, pair):
        e_seq, _, _ = _run_sequential(list(pair))
        e_bat, _, _ = _run_batched(list(pair))
        for i in range(2):
            diff = (e_seq[i] - e_bat[i]).abs().item()
            assert diff < 1e-6, \
                f"pair={pair} mol={i} E_diff={diff:.2e}"

    @pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
    def test_forces(self, pair):
        _, f_seq, _ = _run_sequential(list(pair))
        _, f_bat, _ = _run_batched(list(pair))
        for i in range(2):
            diff = (f_seq[i] - f_bat[i]).abs().max().item()
            assert diff < 1e-5, \
                f"pair={pair} mol={i} F_diff={diff:.2e}"

    @pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
    def test_param_grad(self, pair):
        _, _, g_seq = _run_sequential(list(pair))
        _, _, g_bat = _run_batched(list(pair))
        for i in range(2):
            diff = (g_seq[i] - g_bat[i]).abs().max().item()
            assert diff < 1e-6, \
                f"pair={pair} mol={i} G_diff={diff:.2e}"
