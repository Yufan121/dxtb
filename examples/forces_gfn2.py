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
Example demonstrating force calculations using GFN2-xTB.

This example shows two ways to calculate forces for a molecule:
1. Manually using PyTorch's autograd
2. Using the built-in Calculator.forces() method

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
# dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

# Load molecule data
# f = Path(__file__).parent / "molecules" / "sh3.coord"
# # f = Path(__file__).parent / "molecules" / "h2o.coord"
# numbers, positions = read.read(f, ftype="tm", **dd)
# Xyz file
path = Path(__file__).resolve().parent / "molecules" / "lih.xyz"
numbers, positions = read.read(path, ftype="xyz", **dd)
print(f'numbers: {numbers}, positions: {positions}')

charge = 0

# Calculator options
opts = {
    "verbosity": 0,
    # "fermi_etemp": 300,  # Electronic temperature in K
    # "fermi_maxiter": 500,  # Maximum SCF iterations
    # "scf_mode": "default",  # SCF convergence mode
    # "scp_mode": "fock",  # SCF potential mode
    # if per-atom parameters are used, the default is False
    "per_atom": True
}

######################################################################


print("Calculating forces manually with :func:`torch.autograd.grad`.\n")

time_start = time.time()
dxtb.timer.reset()


glob_param_enum = ['wexp', 'kpol', 'enscale', 'ss', 'pp', 'dd', 'sd', 'pd', 's6', 's8', 'a1', 'a2', 's9', 's10', 'kexp', 'klight', 'gexp', 's', 'p', 'd', 'dmp3', 'dmp5', 'shift', 'rmax']
ele_param_enum = ['levels', 'slater', 'ngauss', 'refocc', 'shpoly', 'kcn', 'gam', 'lgam', 'gam3', 'zeff', 'arep', 'xbond', 'en', 'dkernel', 'qkernel', 'mprad', 'mpvcn']
len_ele_param_enum = [3, 3, 3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pair_param_enum = ['c6matrix']




atom_param_dict = { # arranged by (param, atom). pass to param iniatialization.
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
    "kexp": 0.0,
    "shift": 0.0,
    "rmax": 0.0,
    

    # atoms' parameters
    "levels": [[0,0,0], [0,0,0]],   
    "slater": [[-0.1,-0.2,-0.3], [0,0,0]],   
    # "ngauss": [[0,0,0], [0,0,0]],   # int should be ignored
    # "refocc": [[0,0,0], [0,0,0]],   # int might be ignored
    "shpoly": [[0,0,0], [0,0,0]],   
    "kcn": [[0,0,0], [0,0,0]],
    "gam": [0.1, 0.2],   # Done
    "lgam": [[0.0,0.1,0.2], [0.3,0.4,0.5]],  # Done
    "gam3": [0.5, 0.5],     # Done
    "zeff": [10, 0],    # dF/dp problem
    "arep": [10, 0],    # dF/dp problem
    "en": [0, 0],
    # multipole parameters
    "dkernel": [0, 0],         # Done
    "qkernel": [0, 0],         # Done
    "mprad": [0, 0],          # Done
    "mpvcn": [0, 0],          # Done
    
    # pair parameters
    "c6matrix": [[0,0], [0,0]],
}
# make all values tensors and requires_grad = True
for key, value in atom_param_dict.items():
    if isinstance(value, list) or isinstance(value, float):
        atom_param_dict[key] = torch.tensor(value, dtype=torch.double, requires_grad=True)
    else:
        raise ValueError(f"Invalid param value type for {key}: {type(value)}")
        
# print(atom_param_dict)

# pass the parameters to the calculator
# param_atom = ParamPerAtom.from_dict(atom_param_dict)

GFN2_XTB_ATOM = dxtb.GFN2_XTB

# set it to Param class's vars for access
GFN2_XTB_ATOM._per_atom_params_dict = atom_param_dict

# Initialize calculator with GFN2-xTB and libcint backend
calc = dxtb.Calculator(
    numbers,
    GFN2_XTB_ATOM,  # lazyloaded
    opts=opts,
    **dd
)

# Calculate energy and forces using autograd
pos = positions.clone().requires_grad_(True)
energy = calc.energy(pos, chrg=charge, spin=1)

# Calculate forces as negative gradient of energy
(g,) = torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy), retain_graph=True, create_graph=True)
forces1 = -g

dxtb.timer.print()
time_end = time.time()
print(f"Time taken: {time_end - time_start:.2f} seconds")

######################################################################

print("\n\n\nCalculating forces with Calculator method.\n")

time_start = time.time()
dxtb.timer.reset()

# Reset calculator and calculate forces using built-in method
calc.reset()
pos = positions.clone().requires_grad_(True)
forces2 = calc.forces(pos, chrg=charge)

dxtb.timer.print()
time_end = time.time()
print(f"Time taken: {time_end - time_start:.2f} seconds")

# Verify both methods give identical results
equal = torch.allclose(forces1, forces2, atol=1e-6, rtol=1e-6)
print("\n\nForces are equal:", equal)

# Print some statistics about the forces
print("\nForce statistics:")
print(f"Max force: {forces1.abs().max().item():.6f} Hartree/Bohr")
print(f"Mean force: {forces1.abs().mean().item():.6f} Hartree/Bohr")
print(f"RMS force: {torch.sqrt((forces1**2).mean()).item():.6f} Hartree/Bohr")

######################################################################

print("\n\n\n")
print(f"Calculating dE/dp and dF/dp using torch.autograd.grad")

def get_grad(energy, forces, param):
    grad_energy = torch.autograd.grad(energy, param, grad_outputs=torch.ones_like(energy), retain_graph=True, allow_unused=True)
    grad_forces = torch.autograd.grad(forces, param, grad_outputs=torch.ones_like(forces), retain_graph=True, allow_unused=True)
    
    return grad_energy, grad_forces

for key, value in atom_param_dict.items():
    if isinstance(value, torch.Tensor) and value.requires_grad:
        print(f"{key} is a tensor and requires grad")
        grad_energy, grad_forces = get_grad(energy, forces1, value)
        print(f"dE/dp: {grad_energy}")
        print(f"dF/dp: {grad_forces}")
        print("\n")
    else:
        raise ValueError(f"Invalid param value type for {key}: {type(value)}")

