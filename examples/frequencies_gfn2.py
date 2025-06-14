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
Example demonstrating frequency calculations using GFN2-xTB.

This example shows two ways to calculate vibrational frequencies for a molecule:
1. Using analytical differentiation with manual jacobian
2. Using analytical differentiation with functorch

Both methods should give identical results.
"""
from pathlib import Path

import torch
import time
from tad_mctc.io import read

import dxtb
from dxtb.typing import DD

# Set up device and dtype
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dd: DD = {"device": device, "dtype": torch.double}

# Load molecule data
path = Path(__file__).resolve().parent / "molecules" / "lih.xyz"
numbers, positions = read.read(path, ftype="xyz", **dd)
print(f'numbers: {numbers}, positions: {positions}')

charge = 0

# Calculator options
opts = {
    "verbosity": 0,
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "f_atol": 1e-10,
    "x_atol": 1e-10,
    # if per-atom parameters are used, the default is False
    "per_atom": True
}

######################################################################

# Per-atom parameter setup (similar to forces example)
atom_param_dict = { # arranged by (param, atom). pass to param initialization.
    # global parameters
    "wexp": 0.0,
    "kpol": 0.0,
    "enscale": 0.0,
    "ss": 0.0,
    "pp": 0.0,
    "dd": 0.0,
    "sd": 0.0,
    "pd": 0.0,
    "s6": 0.0,
    "s8": 0.0,
    "a1": 0.0,
    "a2": 0.0,
    "s9": 0.0,
    "s10": 0.0,
    "kexp": 0.0,
    "klight": 0.0,
    "gexp": 0.0,
    "s": 0.0,
    "p": 0.0,
    "d": 0.0,
    "dmp3": 0.0,
    "dmp5": 0.0,
    "shift": 0.0,
    "rmax": 0.0,
    
    # atoms' parameters
    "levels": [[0,0,0], [0,0,0]],   
    "slater": [[-0.1,-0.2,-0.3], [0,0,0]],   
    "shpoly": [[0,0,0], [0,0,0]],   
    "kcn": [[0,0,0], [0,0,0]],
    "gam": [0.1, 0.2],
    "lgam": [[0.0,0.1,0.2], [0.3,0.4,0.5]],
    "gam3": [0.5, 0.5],
    "zeff": [10, 0],
    "arep": [10, 0],
    "en": [0, 0],
    # multipole parameters
    "dkernel": [0, 0],
    "qkernel": [0, 0],
    "mprad": [0, 0],
    "mpvcn": [0, 0],
    
    # pair parameters
    "c6matrix": [[0,0], [0,0]],
}

# make all values tensors and requires_grad = True
for key, value in atom_param_dict.items():
    if isinstance(value, list) or isinstance(value, float):
        atom_param_dict[key] = torch.tensor(value, dtype=torch.double, requires_grad=True)
    else:
        raise ValueError(f"Invalid param value type for {key}: {type(value)}")

GFN2_XTB_ATOM = dxtb.GFN2_XTB

# set it to Param class's vars for access
GFN2_XTB_ATOM._per_atom_params_dict = atom_param_dict

######################################################################

print("Calculating frequencies using analytical differentiation (manual jacobian).\n")

time_start = time.time()
dxtb.timer.reset()

# Initialize calculator with GFN2-xTB
calc = dxtb.Calculator(
    numbers,
    GFN2_XTB_ATOM,
    opts=opts,
    **dd
)

# Calculate frequencies using analytical method
pos = positions.clone().requires_grad_(True)
freqs1, modes1 = calc.vibration(pos, chrg=charge, use_functorch=False)

print(f"Analytical frequencies (manual jacobian) (cm⁻¹): {freqs1}")

dxtb.timer.print()
time_end = time.time()
print(f"Time taken: {time_end - time_start:.2f} seconds")

######################################################################

print("\n\n\nCalculating frequencies using analytical differentiation (functorch).\n")

time_start = time.time()
dxtb.timer.reset()

# Reset calculator and calculate frequencies using functorch
calc.reset()
pos = positions.clone().requires_grad_(True)
freqs2, modes2 = calc.vibration(pos, chrg=charge, use_functorch=True)

print(f"Analytical frequencies (functorch) (cm⁻¹): {freqs2}")

dxtb.timer.print()
time_end = time.time()
print(f"Time taken: {time_end - time_start:.2f} seconds")

######################################################################

# Compare results
print("\n\nComparing results:")
print(f"Manual jacobian:  {freqs1}")
print(f"Functorch:        {freqs2}")

# Check if analytical methods agree
equal_analytical = torch.allclose(freqs1, freqs2, atol=1e-6, rtol=1e-6)
print(f"\nAnalytical methods agree: {equal_analytical}")

# Print frequency statistics
print("\nFrequency statistics:")
print(f"Number of frequencies: {len(freqs1)}")
print(f"Lowest frequency: {freqs1.min().item():.2f} cm⁻¹")
print(f"Highest frequency: {freqs1.max().item():.2f} cm⁻¹")

######################################################################

print("\n\n\n")
print(f"Calculating dFreq/dp using torch.autograd.grad")

def get_freq_grad(freqs, param):
    grad_freqs = torch.autograd.grad(
        freqs, param, 
        grad_outputs=torch.ones_like(freqs), 
        retain_graph=True, 
        allow_unused=True
    )
    return grad_freqs

# Calculate frequency gradients with respect to parameters
for key, value in atom_param_dict.items():
    if isinstance(value, torch.Tensor) and value.requires_grad:
        print(f"{key} is a tensor and requires grad")
        grad_freqs = get_freq_grad(freqs1, value)
        print(f"dFreq/dp: {grad_freqs}")
        print("\n")
    else:
        raise ValueError(f"Invalid param value type for {key}: {type(value)}") 