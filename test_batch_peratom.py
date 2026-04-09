"""
Test: batched per-atom parameter correction vs sequential single-molecule.

Success criteria:
  1. E_batch[i] == E_seq[i]  (energy match)
  2. F_batch[i] == F_seq[i]  (force match, F = -dE/dR)
  3. dE/dp_batch == dE/dp_seq  (parameter gradient match)
"""

import torch
import dxtb
import tomli as toml
from dxtb._src.param.base import Param
from tad_mctc.batch import pack
from pathlib import Path

torch.set_default_dtype(torch.float64)

ANG2BOHR = 1.8897261328856432

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

def load_base_param():
    global _CACHED_TOML
    if _CACHED_TOML is None:
        toml_path = Path(dxtb.__file__).parent / "_src" / "param" / "gfn2" / "gfn2-xtb.toml"
        with open(toml_path, "rb") as fd:
            _CACHED_TOML = toml.load(fd)
    return Param(**_CACHED_TOML)


def make_full_params(natom, small=0.01, seed=42):
    """Create per-atom correction dict for a single molecule."""
    torch.manual_seed(seed)
    params = {}
    for key, length in ELEM_PARAMS.items():
        shape = (natom,) if length == 1 else (natom, length)
        params[key] = (torch.randn(shape) * small).requires_grad_(True)
    for key in PAIR_PARAMS:
        params[key] = (torch.randn(natom, natom) * small).requires_grad_(True)
    return params


def make_batched_params(per_mol_params, max_nat):
    """Pack per-molecule param dicts into batched tensors.

    Each per-atom tensor gets zero-padded to max_nat along the atom dim.
    Returns a new dict of (batch, max_nat, ...) tensors that are LEAF tensors
    with requires_grad=True. Also returns a list of original per-mol tensors
    so we can compare gradients later.
    """
    batch = len(per_mol_params)
    batched = {}

    for key in per_mol_params[0]:
        tensors = [p[key] for p in per_mol_params]
        # Pack with zero padding
        batched[key] = pack(
            [t.detach() for t in tensors], value=0.0
        ).requires_grad_(True)

    return batched


MOLECULES = [
    ("H2O", torch.tensor([8, 1, 1]),
     torch.tensor([
         [0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692],
     ]) * ANG2BOHR),
    ("CH4", torch.tensor([6, 1, 1, 1, 1]),
     torch.tensor([
         [0.0, 0.0, 0.0], [0.6276, 0.6276, 0.6276],
         [0.6276, -0.6276, -0.6276], [-0.6276, 0.6276, -0.6276],
         [-0.6276, -0.6276, 0.6276],
     ]) * ANG2BOHR),
    ("LiH", torch.tensor([3, 1]),
     torch.tensor([
         [0.0, 0.0, 0.0], [0.0, 0.0, 1.596],
     ]) * ANG2BOHR),
]


def run_sequential(mol_indices):
    """Run molecules one-by-one, return (energies, forces, param_grads)."""
    energies = []
    forces_list = []
    param_grads = []

    for idx in mol_indices:
        name, numbers, positions = MOLECULES[idx]
        positions = positions.clone().requires_grad_(True)
        params = make_full_params(len(numbers), seed=100 + idx)

        par = load_base_param()
        par._per_atom_params_dict = params
        calc = dxtb.Calculator(numbers, par, opts={"verbosity": 0, "batch_mode": 0},
                               dtype=torch.float64)
        e = calc.energy(positions)
        energies.append(e.detach())

        (neg_f,) = torch.autograd.grad(e, positions, retain_graph=True)
        forces_list.append((-neg_f).detach())

        # dE/dp for "en" param
        (g,) = torch.autograd.grad(e, params["en"], retain_graph=True)
        param_grads.append(g.detach())

    return energies, forces_list, param_grads


def run_batched(mol_indices):
    """Run molecules as a padded batch, return (energies, forces, param_grads)."""
    mols = [MOLECULES[i] for i in mol_indices]
    per_mol_params = [make_full_params(len(m[1]), seed=100 + i) for i, m in zip(mol_indices, mols)]

    # Batch numbers and positions
    numbers_batch = pack([m[1] for m in mols], value=0)
    positions_batch = pack(
        [m[2].clone() for m in mols], value=0.0
    ).requires_grad_(True)

    # Batch per-atom params
    batched_params = make_batched_params(per_mol_params, numbers_batch.shape[-1])

    # Create batched calculator
    par = load_base_param()
    par._per_atom_params_dict = batched_params
    chrg = torch.zeros(len(mols), dtype=torch.float64)

    calc = dxtb.Calculator(numbers_batch, par,
                           opts={"verbosity": 0, "batch_mode": 1},
                           dtype=torch.float64)
    e = calc.energy(positions_batch, chrg=chrg)

    # Forces
    (neg_f,) = torch.autograd.grad(e.sum(), positions_batch, retain_graph=True)
    forces_batch = -neg_f

    # dE/dp for "en" param
    (g,) = torch.autograd.grad(e.sum(), batched_params["en"], retain_graph=True)

    # Unpack results
    energies = [e[i].detach() for i in range(len(mols))]
    natoms = [len(m[1]) for m in mols]
    forces_list = [forces_batch[i, :natoms[i]].detach() for i in range(len(mols))]
    param_grads = [g[i, :natoms[i]].detach() for i in range(len(mols))]

    return energies, forces_list, param_grads


def test_batch_vs_sequential(mol_indices, label=""):
    """Compare batch vs sequential results."""
    print(f"=== Test: {label} ===")
    names = [MOLECULES[i][0] for i in mol_indices]
    print(f"  Molecules: {names}")

    e_seq, f_seq, g_seq = run_sequential(mol_indices)
    e_bat, f_bat, g_bat = run_batched(mol_indices)

    all_pass = True
    for i, idx in enumerate(mol_indices):
        name = MOLECULES[idx][0]

        # Energy
        e_match = torch.allclose(e_seq[i], e_bat[i], atol=1e-6)
        e_diff = (e_seq[i] - e_bat[i]).abs().item()

        # Forces
        f_match = torch.allclose(f_seq[i], f_bat[i], atol=1e-5)
        f_diff = (f_seq[i] - f_bat[i]).abs().max().item()

        # dE/dp
        g_match = torch.allclose(g_seq[i], g_bat[i], atol=1e-6)
        g_diff = (g_seq[i] - g_bat[i]).abs().max().item()

        status = "OK" if (e_match and f_match and g_match) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {name}: E_diff={e_diff:.2e} F_diff={f_diff:.2e} G_diff={g_diff:.2e} [{status}]")
        if not e_match:
            print(f"    E_seq={e_seq[i].item():.10f}  E_bat={e_bat[i].item():.10f}")
        if not f_match:
            print(f"    F_seq={f_seq[i]}")
            print(f"    F_bat={f_bat[i]}")
        if not g_match:
            print(f"    G_seq={g_seq[i]}")
            print(f"    G_bat={g_bat[i]}")

    print(f"  {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


if __name__ == "__main__":
    ok = True

    # Same molecule batch (easiest)
    ok &= test_batch_vs_sequential([0, 0], "2x H2O (same)")

    # Different molecules
    ok &= test_batch_vs_sequential([0, 1], "H2O + CH4 (different)")

    # Three different molecules
    ok &= test_batch_vs_sequential([0, 1, 2], "H2O + CH4 + LiH (all different)")

    if ok:
        print("All batch vs sequential tests PASSED!")
    else:
        print("Some tests FAILED!")
        exit(1)
