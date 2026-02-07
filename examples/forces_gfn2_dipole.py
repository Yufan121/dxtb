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
import random
import torch
import time
from tad_mctc.io import read

import dxtb
from dxtb.typing import DD
import numpy as np
from dxtb._src.components.interactions import new_efield

import json

# torch.set_num_interop_threads(4)

# Set up device and dtype
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dd: DD = {"device": device, "dtype": torch.double}
# dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

# Load molecule data
# f = Path(__file__).parent / "molecules" / "sh3.coord"
# # f = Path(__file__).parent / "molecules" / "h2o.coord"
# numbers, positions = read.read(f, ftype="tm", **dd)
# Xyz file
# path = Path(__file__).resolve().parent / "molecules" / "capsaicin.xyz"
path = Path(__file__).resolve().parent / "molecules" / "FH-BH2.xyz"
numbers, positions = read.read(path, ftype="xyz", **dd)
# print(f'numbers: {numbers}, positions: {positions}')


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

    ###### atoms' parameters
    "levels": [[0,0,0], [0,0,0]],   
    "slater": [[0.0,0.0,0.0], [0.0,0.0,0.0]],   
    # "ngauss": [[0,0,0], [0,0,0]],   # int should be ignored
    # "refocc": [[0,0,0], [0,0,0]],   # int might be ignored
    "shpoly": [[0,0,0], [0,0,0]],   
    "kcn": [[0,0,0], [0,0,0]],
    "gam": [0.0, 0.0],   # Done
    "lgam": [[0.0,0.0,0.0], [0.0,0.0,0.0]],  # Done
    "gam3": [0.0, 0.0],     # Done
    "zeff": [0, 0],    # dF/dp problem
    "arep": [0, 0],    # dF/dp problem
    "en": [0, 0],
    # multipole parameters
    "dkernel": [0, 0],         # Done
    "qkernel": [0, 0],         # Done
    "mprad": [0, 0],          # Done
    "mpvcn": [0, 0],          # Done
    "3rd_scale": [[0, 0, 0], [0, 0, 0]],
    # "qsh": [[1, 2, 3], [4, 5, 6]],
    # "predicted_energy": [[1, 2, 3], [4, 5, 6]],
    "rcov": [0, 0],
    "arad": [0, 0],
    
    
}

# copy each value in atom_param_dict to match numbers
natom = len(numbers)
for key, value in atom_param_dict.items():
    if isinstance(value, list):
        atom_param_dict[key] = [value[0]] * natom 
    else:
        atom_param_dict[key] = value


# from params_dict.json, load the params_dict
with open(Path(__file__).resolve().parent / "params_dict.json", "r") as f:
    params_dict = json.load(f)
    atom_param_dict = params_dict["params_dict"]
print(atom_param_dict)

# make all values tensors and requires_grad = True
for key, value in atom_param_dict.items():
    if isinstance(value, list) or isinstance(value, float):
        atom_param_dict[key] = torch.tensor(value, dtype=torch.double, requires_grad=True) 
        # if isinstance(value, list): # list
        #     atom_param_dict[key] = atom_param_dict[key] + torch.randn(1) * 0.1
        # else: # float
        #     atom_param_dict[key] = atom_param_dict[key] + torch.randn(1) * 0.1
    else:
        raise ValueError(f"Invalid param value type for {key}: {type(value)}")
        
# print(atom_param_dict)

# pass the parameters to the calculator
# param_atom = ParamPerAtom.from_dict(atom_param_dict)

GFN2_XTB_ATOM = dxtb.GFN2_XTB

# set it to Param class's vars for access
GFN2_XTB_ATOM._per_atom_params_dict = atom_param_dict

# --- Correct initialization for dipole calculation ---

# Set up a differentiable electric field vector
efield_vec = torch.zeros(3, dtype=torch.double, device=device, requires_grad=True) 
print(f"\nefield_vec: {efield_vec}")
efield = new_efield(efield_vec)

# Initialize calculator with GFN2-xTB, libcint backend, and electric field
calc = dxtb.Calculator(
    numbers,
    GFN2_XTB_ATOM,  # lazyloaded
    interaction=[efield],
    opts=opts,
    **dd
)

# Calculate energy and forces using autograd
charge = 0
spin = 0

pos = positions.clone().requires_grad_(True)
energy = calc.energy(pos, chrg=charge, spin=spin)

# Calculate forces as negative gradient of energy
(g,) = torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy), retain_graph=True, create_graph=True)
forces1 = -g
print(f"forces1: {forces1}")

######################################################################

print("Calculating dipole moment using autograd (manual Jacobian).\n")

# Compute dipole using the Calculator's dipole method (autograd, manual Jacobian)
dipole_autograd = calc.dipole(pos, chrg=charge, spin=spin, use_functorch=False)
print(f"Dipole (autograd, manual Jacobian): {dipole_autograd.detach().cpu().numpy()}, magnitude: {np.linalg.norm(dipole_autograd.detach().cpu().numpy())}")
au2debye = 2.5417464
print(f"Magnitude of dipole (Debye): {np.linalg.norm(dipole_autograd.detach().cpu().numpy()) * au2debye}")

# # Optionally, also compute dipole using functorch Jacobian
# dipole_functorch = calc.dipole(pos, chrg=charge, spin=spin, use_functorch=True)
# print(f"Dipole (autograd, functorch): {dipole_functorch.detach().cpu().numpy()}")

dxtb.timer.print()
time_end = time.time()
print(f"Time taken: {time_end - time_start:.2f} seconds")
print(f"Energy: {energy.item():.6f} Hartree")

# if energy is nan, print the atom_param_dict
if torch.isnan(energy).any():
    print(f"atom_param_dict: {atom_param_dict}")
    raise ValueError("Energy is nan")
