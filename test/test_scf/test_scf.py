"""
Test for SCF.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""

import math

import numpy as np
import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from ..utils import load_from_npz
from .samples import samples

# torch.autograd.set_detect_anomaly(True)
opts = {"verbosity": 0, "etemp": 300.0, "guess": "eeq"}

ref_grad = np.load("test/test_scf/grad.npz")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "CH4", "SiH4"])
def test_single(dtype: torch.dtype, name: str):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"].item()
    charges = torch.tensor(0.0).type(dtype)

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1).item()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize(
    "name", ["S2", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01", "LYS_xao", "C60"]
)
def test_single2(dtype: torch.dtype, name: str):
    """Test a few larger system (only float32 as they take some time)."""
    tol = math.sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"].item()
    charges = torch.tensor(0.0).type(dtype)

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1).item()


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["vancoh2"])
def test_single_large(dtype: torch.dtype, name: str):
    """Test a large systems (only float32 as they take some time)."""
    tol = math.sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"]
    charges = torch.tensor(0.0).type(dtype)

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1).item()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample[0]["numbers"],
            sample[1]["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample[0]["positions"],
            sample[1]["positions"],
        )
    ).type(dtype)
    ref = batch.pack(
        (
            sample[0]["escf"],
            sample[1]["escf"],
        )
    ).type(dtype)
    charges = torch.tensor([0.0, 0.0], **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert torch.allclose(ref, result.scf.sum(-1), atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["LiH"])
def test_grad_backwards(name: str, dtype: torch.dtype = torch.float):
    # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
    tol = math.sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    pos = samples[name]["positions"].type(dtype)
    positions = pos.detach().clone().requires_grad_(True)
    charges = torch.tensor(0.0, **dd)
    ref = load_from_npz(ref_grad, name, dtype)

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    energy.backward()
    if positions.grad is None:
        assert False
    gradient = positions.grad.clone()
    assert torch.allclose(gradient, ref, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "CH4", "SiH4"])
def test_grad(name: str, dtype: torch.dtype):
    # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
    tol = math.sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)
    ref = load_from_npz(ref_grad, name, dtype)

    op = dict(opts, **{"exclude": ["rep", "disp", "hal"]})
    calc = Calculator(numbers, par, opts=op, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
    )[0]

    assert torch.allclose(gradient, ref, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["LYS_xao", "C60", "vancoh2"])
def test_grad_large(name: str, dtype: torch.dtype = torch.float):
    # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
    tol = math.sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)
    ref = load_from_npz(ref_grad, name, dtype)

    op = dict(opts, **{"exclude": ["rep", "disp", "hal"]})
    calc = Calculator(numbers, par, opts=op, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
    )[0]

    print(gradient)

    assert torch.allclose(gradient, ref, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize(
    "testcase",
    [
        (
            "LiH",
            {
                "selfenergy": torch.tensor(
                    [+0.0002029369, +0.0017547115, +0.1379896402, -0.1265652627]
                ),
                "kcn": torch.tensor(
                    [-0.1432282478, -0.0013212233, -0.1811404824, +0.0755317509]
                ),
                "shpoly": torch.tensor(
                    [+0.0408593193, -0.0007219329, -0.0385218151, +0.0689999014]
                ),
            },
        ),
    ],
)
def test_gradgrad(testcase, dtype: torch.dtype = torch.float):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    name, ref = testcase
    numbers = samples[name]["numbers"]
    pos = samples[name]["positions"].type(dtype)
    positions = pos.detach().clone().requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    calc.hamiltonian.selfenergy.requires_grad_(True)
    calc.hamiltonian.kcn.requires_grad_(True)
    calc.hamiltonian.shpoly.requires_grad_(True)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
        create_graph=True,
    )[0]

    pgrad = torch.autograd.grad(
        gradient[0, :].sum(),
        (calc.hamiltonian.selfenergy, calc.hamiltonian.kcn, calc.hamiltonian.shpoly),
    )

    assert torch.allclose(pgrad[0], ref["selfenergy"], atol=tol)
    assert torch.allclose(pgrad[1], ref["kcn"], atol=tol)
    assert torch.allclose(pgrad[2], ref["shpoly"], atol=tol)
