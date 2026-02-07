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

from typing import Tuple, Union

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
path = Path(__file__).resolve().parent / "molecules" / "test.xyz"
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
    # "per_atom": True
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
    
    ##### pair parameters
    # "c6matrix": [[0,0], [0,0]],
    # "theta_ss": [[0.1,0.2], [0.3,0.4]],
    # "theta_pp": [[0.5,0.6], [0.7,0.8]],
    # "theta_dd": [[0.9,0.10], [0.11,0.12]],
    # "theta_sp": [[0.13,0.14], [0.15,0.16]],
    # "theta_sd": [[0.17,0.18], [0.19,0.20]],
    # "theta_pd": [[0.21,0.22], [0.23,0.24]],
    
    # "zeta_ss": [[0,0], [0,0]],
    # "zeta_pp": [[0,0], [0,0]],
    # "zeta_dd": [[0,0], [0,0]],
    # "zeta_sp": [[0,0], [0,0]],
    # "zeta_sd": [[0,0], [0,0]],
    # "zeta_pd": [[0,0], [0,0]],
    
}

# copy each value in atom_param_dict to match numbers
natom = len(numbers)
for key, value in atom_param_dict.items():
    if isinstance(value, list):
        atom_param_dict[key] = [value[0]] * natom
    else:
        atom_param_dict[key] = value

# # randomly perturb only a subset of keys, and save the list of keys to a variable
# import random

# upper = 1
# lower = -1

# all_keys = list(atom_param_dict.keys()) # random candidates
# # all_keys =  ['lgam', 'mprad', 'gexp', 'a1', 'shpoly']
# # keys_to_perturb = random.sample(all_keys, k=int(len(all_keys) * 0.5))  # perturb 50% of keys
# keys_to_perturb = list(atom_param_dict.keys())  # all params

# for key in keys_to_perturb:
#     value = atom_param_dict[key]
#     if isinstance(value, list):
#         if all(isinstance(i, list) for i in value):  # Check for two-dimensional array
#             atom_param_dict[key] = [[random.uniform(lower, upper) for _ in range(len(sublist))] for sublist in value]
#         else:
#             atom_param_dict[key] = [random.uniform(lower, upper) for _ in range(len(value))]
#     else:
#         atom_param_dict[key] = random.uniform(lower, upper)


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
charge = 1
spin = 0

charge = torch.tensor(charge, dtype=torch.double, requires_grad=True)
spin = torch.tensor(spin, dtype=torch.double, requires_grad=True)

pos = positions.clone().requires_grad_(True)
energy = calc.energy(pos, chrg=charge, spin=spin)

# Calculate forces as negative gradient of energy
(g,) = torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy), retain_graph=True, create_graph=True)
forces1 = -g


dxtb.timer.print()
time_end = time.time()
print(f"Time taken: {time_end - time_start:.2f} seconds")
print(f"Energy: {energy.item():.6f} Hartree")

# if energy is nan, print the atom_param_dict
if torch.isnan(energy).any():
    print(f"atom_param_dict: {atom_param_dict}")
    print(f"keys_to_perturb: {keys_to_perturb}")
    raise ValueError("Energy is nan")

######################################################################

print("\n\n\nCalculating forces with Calculator method.\n")

time_start = time.time()
dxtb.timer.reset()

# Reset calculator and calculate forces using built-in method
calc.reset()
pos = positions.clone().requires_grad_(True)
forces2 = calc.forces_numerical(pos, chrg=charge)

dxtb.timer.print()
time_end = time.time()
print(f"Time taken: {time_end - time_start:.2f} seconds")

# Verify both methods give identical results
equal = torch.allclose(forces1, forces2, atol=1e-6, rtol=1e-6)
print("\n\nForces are equal:", equal)

# Print diff between force 1 and force 2
diff = forces1 - forces2
print(f"Diff between force 1 and force 2: {diff}")
print(f"Max diff: {diff.abs().max().item():.6f} Hartree/Bohr")
print(f"Mean diff: {diff.abs().mean().item():.6f} Hartree/Bohr")
print(f"RMS diff: {torch.sqrt((diff**2).mean()).item():.6f} Hartree/Bohr")

from dxtb.calculators import AutogradCalculator

print(AutogradCalculator.implemented_properties)


# ========== Compute frequencies using dxtb analytic vibration module ==========

from typing import Union, Tuple

def compute_frequencies_dxtb(calc, 
                            positions: torch.Tensor, 
                            charge: int = 0,
                            return_modes: bool = False,
                            convert_units: bool = True,
                            cut_freqs: bool = False
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    计算频率（使用解析方法）

    Args:
        calc: dXTB计算器
        positions: 原子位置 (已经是正确的设备和数据类型)
        charge: 电荷
        return_modes: 是否返回振动模式
        convert_units: 是否转换单位到 cm⁻¹
        cut_freqs: 是否只保留头3和尾6个频率

    Returns:
        频率张量 (如果 return_modes=False) 或 (频率张量, 振动模式张量) 的元组
    """
    # 确保位置张量需要梯度
    pos = positions.requires_grad_(True)

    # 使用解析方法计算频率和振动模式 (manual jacobian)
    freqs, modes = calc.vibration(pos, chrg=charge, use_functorch=False)

    # 转换单位：原子单位 -> cm⁻¹
    if convert_units:
        try:
            from tad_mctc.units import AU2RCM
            freqs_cm = freqs * AU2RCM
            freqs = freqs_cm
        except ImportError:
            print("Warning: tad_mctc.units not available, using conversion factor")
            # 手动转换因子
            AU2RCM = 5140.487
            freqs_cm = freqs * AU2RCM
            print(f"Analytical frequencies (manual jacobian) (atomic units): {freqs}")
            print(f"Analytical frequencies (manual jacobian) (cm⁻¹): {freqs_cm}")
            freqs = freqs_cm

    if cut_freqs:
        # Show only the first 3 and last 6 frequencies
        if freqs.shape[0] > 9:
            freqs = torch.cat((freqs[:3], freqs[-6:]))
        else:
            print("Warning: Not enough frequencies to cut, showing all.")

    if return_modes:
        return freqs, modes
    else:
        return freqs

def compute_frequencies_dxtb_numerical(calc, 
                                       positions: torch.Tensor, 
                                       charge: int = 0,
                                       return_modes: bool = False,
                                       convert_units: bool = True,
                                       cut_freqs: bool = False
                                       ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    计算频率（使用数值方法）

    Args:
        calc: dXTB计算器
        positions: 原子位置 (已经是正确的设备和数据类型)
        charge: 电荷
        return_modes: 是否返回振动模式
        convert_units: 是否转换单位到 cm⁻¹
        cut_freqs: 是否只保留头3和尾6个频率

    Returns:
        频率张量 (如果 return_modes=False) 或 (频率张量, 振动模式张量) 的元组
    """
    pos = positions.requires_grad_(False).clone().detach()
    freqs, modes = calc.vibration_numerical(pos, chrg=charge)

    if convert_units:
        try:
            from tad_mctc.units import AU2RCM
            freqs_cm = freqs * AU2RCM
            freqs = freqs_cm
        except ImportError:
            print("Warning: tad_mctc.units not available, using conversion factor")
            AU2RCM = 5140.487
            freqs_cm = freqs * AU2RCM
            print(f"Numerical frequencies (atomic units): {freqs}")
            print(f"Numerical frequencies (cm⁻¹): {freqs_cm}")
            freqs = freqs_cm

    if cut_freqs:
        if freqs.shape[0] > 9:
            freqs = torch.cat((freqs[:3], freqs[-6:]))
        else:
            print("Warning: Not enough frequencies to cut, showing all.")

    if return_modes:
        return freqs, modes
    else:
        return freqs

print("\n\nCalculating vibrational frequencies (analytic)...\n")
freqs_analytic = compute_frequencies_dxtb(
    calc, positions, charge=charge, return_modes=False, convert_units=True, cut_freqs=False
)
print("All vibrational frequencies (analytic, cm⁻¹):", freqs_analytic.cpu().detach().numpy())

print("\nCalculating vibrational frequencies (numerical)...\n")
freqs_numerical = compute_frequencies_dxtb_numerical(
    calc, positions, charge=charge, return_modes=False, convert_units=True, cut_freqs=False
)
print("All vibrational frequencies (numerical, cm⁻¹):", freqs_numerical.cpu().detach().numpy())

# Optionally, print the diff (analytic - numerical)
if freqs_analytic.shape == freqs_numerical.shape:
    diff = freqs_analytic - freqs_numerical
    print("\nDifference (analytic - numerical):")
    print("Max diff: {:.6f} cm⁻¹".format(diff.abs().max().item()))
    print("Mean diff: {:.6f} cm⁻¹".format(diff.abs().mean().item()))
    print("RMS diff: {:.6f} cm⁻¹".format(torch.sqrt((diff**2).mean()).item()))
    print("Frequency-by-frequency diff:", diff.cpu().detach().numpy())
else:
    print("Warning: Analytic and numerical frequencies have different shapes; can't compute diff.")



