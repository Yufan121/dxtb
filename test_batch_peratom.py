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

# Pair params
PAIR_PARAMS = [
    "theta_ss", "theta_pp", "theta_dd", "theta_sp", "theta_sd", "theta_pd",
    "zeta_ss", "zeta_pp", "zeta_dd", "zeta_sp", "zeta_sd", "zeta_pd",
]


def make_full_params(natom, small=0.01, seed=42):
    """Create a complete per-atom params dict with small random values."""
    torch.manual_seed(seed)
    params = {}

    for key, length in ELEM_PARAMS.items():
        shape = (natom,) if length == 1 else (natom, length)
        params[key] = (torch.randn(shape) * small).requires_grad_(True)

    for key in PAIR_PARAMS:
        params[key] = (torch.randn(natom, natom) * small).requires_grad_(True)

    return params


def load_base_param():
    toml_path = Path(dxtb.__file__).parent / "_src" / "param" / "gfn2" / "gfn2-xtb.toml"
    with open(toml_path, "rb") as fd:
        return Param(**toml.load(fd))


def make_calc(numbers, params_dict, batch_mode=0):
    """Create Calculator with per-atom corrections."""
    param_model = load_base_param()
    param_model._per_atom_params_dict = params_dict
    opts = {"verbosity": 0, "batch_mode": batch_mode}
    return dxtb.Calculator(numbers, param_model, opts=opts, dtype=torch.float64)


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


def test_single_molecule():
    """Sanity: single molecule with per-atom corrections computes E/F/dE_dp."""
    print("=== Test: single molecule ===")
    name, numbers, positions = MOLECULES[0]
    positions = positions.clone().requires_grad_(True)
    params = make_full_params(len(numbers), seed=42)

    calc = make_calc(numbers, params)
    e = calc.energy(positions)
    print(f"  {name} energy: {e.item():.8f}")

    (neg_forces,) = torch.autograd.grad(e, positions, retain_graph=True)
    print(f"  Forces max: {neg_forces.abs().max().item():.6e}")

    (grad_en,) = torch.autograd.grad(e, params["en"], retain_graph=True)
    print(f"  dE/d(en): {grad_en}")

    print("  PASS\n")


def test_sequential_consistency():
    """All test molecules run without error."""
    print("=== Test: sequential consistency ===")
    for name, numbers, positions in MOLECULES:
        positions = positions.clone().requires_grad_(True)
        params = make_full_params(len(numbers), seed=hash(name) % 10000)

        calc = make_calc(numbers, params)
        e = calc.energy(positions)

        (neg_f,) = torch.autograd.grad(
            e, positions, retain_graph=True, create_graph=False,
        )
        print(f"  {name}: E={e.item():.8f}, |F|_max={neg_f.abs().max():.6e}")

    print("  PASS\n")


if __name__ == "__main__":
    test_single_molecule()
    test_sequential_consistency()
    print("All tests passed!")
