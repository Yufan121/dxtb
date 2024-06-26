# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing dispersion gradient (autodiff).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb._src.components.classicals.dispersion import new_dispersion
from dxtb._src.typing import DD, Callable, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01", "PbH4-BiH3"]

tol = 1e-8


def gradchecker(dtype: torch.dtype, name: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)

    def func(positions: Tensor) -> Tensor:
        return disp.get_energy(positions, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(dtype: torch.dtype, name1: str, name2: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)

    def func(positions: Tensor) -> Tensor:
        return disp.get_energy(positions, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    positions.requires_grad_(True)
    ref = sample["grad"].to(**dd)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)

    energy = disp.get_energy(positions, cache)
    grad_autograd = disp.get_gradient(energy, positions)

    assert pytest.approx(ref.cpu(), abs=tol) == grad_autograd.detach().cpu()

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_autograd_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    ref = pack(
        [
            sample1["grad"].to(**dd),
            sample2["grad"].to(**dd),
        ]
    )

    positions.requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)

    energy = disp.get_energy(positions, cache)
    grad_autograd = disp.get_gradient(energy, positions)

    positions.detach_()
    grad_autograd.detach_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_autograd.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_backward(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["grad"].to(**dd)

    # variable to be differentiated (clone in backward for safety)
    pos = positions.clone().requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)

    # automatic gradient
    energy = disp.get_energy(pos, cache)
    energy.sum().backward()

    assert pos.grad is not None
    grad_backward = pos.grad.clone()

    # also zero out gradients when using `.backward()`
    pos.detach_()
    pos.grad.data.zero_()
    grad_backward.detach_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_backward.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_backward_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    ref = pack(
        [
            sample1["grad"].to(**dd),
            sample2["grad"].to(**dd),
        ]
    )

    # variable to be differentiated (clone in backward for safety)
    pos = positions.clone().requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)

    # automatic gradient
    energy = disp.get_energy(pos, cache)
    energy.sum().backward()

    assert pos.grad is not None
    grad_backward = pos.grad.clone()

    # also zero out gradients when using `.backward()`
    pos.detach_()
    pos.grad.data.zero_()
    grad_backward.detach_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_backward.cpu()
