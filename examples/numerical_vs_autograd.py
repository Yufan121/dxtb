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
Example comparing numerical vs autograd results for Energy, Forces, and Frequencies.

This script computes and compares:
1. Energy values
2. Forces (gradient of energy w.r.t. positions)
3. Vibrational frequencies

Both numerical (5-point finite difference) and autograd methods are used.

The 5-point method provides high accuracy numerical derivatives:
f'(x) ≈ [f(x-2h) - 8*f(x-h) + 8*f(x+h) - f(x+2h)] / (12*h)
"""
from pathlib import Path
import torch
import time
import numpy as np
from typing import Union, Tuple
from tad_mctc.io import read

import dxtb
from dxtb.typing import DD


##############################################################################
# Frequency Computation Function
##############################################################################

def compute_frequencies_dxtb(calc: dxtb.Calculator, 
                            positions: torch.Tensor, 
                            charge: int = 0,
                            return_modes: bool = False,
                            convert_units: bool = True,
                            cut_freqs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    计算频率（使用解析方法）
    
    Args:
        calc: dXTB计算器
        positions: 原子位置 (已经是正确的设备和数据类型)
        charge: 电荷
        return_modes: 是否返回振动模式
        convert_units: 是否转换单位到 cm⁻¹
        
    Returns:
        频率张量 (如果 return_modes=False) 或 (频率张量, 振动模式张量) 的元组
    """
    # 确保位置张量在正确的设备上并需要梯度
    # device = positions.device
    # pos = positions.clone().requires_grad_(True).to(device)  # clone保证梯度连接 !!!Clone 并不能保证梯度链接
    pos = positions
    
    # 使用解析方法计算频率和振动模式 (manual jacobian)
    freqs, modes = calc.vibration(pos, chrg=charge, use_functorch=False)
    
    # 转换单位：原子单位 -> cm⁻¹
    if convert_units:
        try:
            from tad_mctc.units import AU2RCM
            freqs_cm = freqs * AU2RCM
            # print(f"Analytical frequencies (manual jacobian) (atomic units): {freqs}")
            # print(f"Analytical frequencies (manual jacobian) (cm⁻¹): {freqs_cm}")
            freqs = freqs_cm
        except ImportError:
            print("Warning: tad_mctc.units not available, using conversion factor")
            # 手动转换因子：1 Hartree = 219474.63 cm⁻¹
            # 但频率的转换需要 sqrt(Hartree/u/bohr²) 到 cm⁻¹
            # 标准转换因子约为 5140.487 cm⁻¹
            AU2RCM = 5140.487  # 近似转换因子
            freqs_cm = freqs * AU2RCM
            print(f"Analytical frequencies (manual jacobian) (atomic units): {freqs}")
            print(f"Analytical frequencies (manual jacobian) (cm⁻¹): {freqs_cm}")
            freqs = freqs_cm
    

    if cut_freqs:
        freqs = torch.cat((freqs[:3], freqs[-6:]))
    
    if return_modes:
        return freqs, modes
    else:
        return freqs


##############################################################################
# Energy, Forces, and Frequencies Computation
##############################################################################

def compute_energy_autograd(calc, positions, charge, spin):
    """Compute energy using autograd."""
    return calc.energy(positions, chrg=charge, spin=spin)


def compute_forces_autograd(calc, positions, charge, spin):
    """Compute forces using autograd."""
    pos = positions.clone().requires_grad_(True)
    energy = calc.energy(pos, chrg=charge, spin=spin)
    (g,) = torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy), 
                                retain_graph=False, create_graph=False)
    forces = -g
    return forces


def compute_energy_numerical(numbers, positions, charge, spin, opts, dd, atom_param_dict, h=1e-5):
    """Compute energy (baseline for numerical methods)."""
    GFN2_XTB = dxtb.GFN2_XTB
    GFN2_XTB._per_atom_params_dict = atom_param_dict
    
    calc = dxtb.Calculator(numbers, GFN2_XTB, opts=opts, **dd)
    with torch.no_grad():
        energy = calc.energy(positions, chrg=charge, spin=spin)
    return energy


def compute_forces_numerical(numbers, positions, charge, spin, opts, dd, atom_param_dict, h=1e-5):
    """Compute forces using 5-point numerical differentiation."""
    natoms = positions.shape[0]
    ndim = positions.shape[1]
    forces = torch.zeros_like(positions)
    
    # Coefficients for 5-point formula
    coeffs = torch.tensor([1.0, -8.0, 8.0, -1.0])
    offsets = torch.tensor([-2, -1, 1, 2])
    
    # Calculate force for each atom and dimension
    for i in range(natoms):
        for j in range(ndim):
            energies = []
            
            for offset, coeff in zip(offsets, coeffs):
                # Perturb position
                pos_perturbed = positions.clone()
                pos_perturbed[i, j] = pos_perturbed[i, j] + offset * h
                
                # Create fresh calculator
                GFN2_XTB = dxtb.GFN2_XTB
                GFN2_XTB._per_atom_params_dict = atom_param_dict
                
                calc = dxtb.Calculator(numbers, GFN2_XTB, opts=opts, **dd)
                
                with torch.no_grad():
                    energy = calc.energy(pos_perturbed, chrg=charge, spin=spin)
                energies.append(coeff * energy.item())
            
            # Apply 5-point formula (negative gradient for forces)
            forces[i, j] = -sum(energies) / (12.0 * h)
    
    return forces


def compute_frequencies_numerical(numbers, positions, charge, spin, opts, dd, atom_param_dict, h=1e-5):
    """
    Compute frequencies using numerical Hessian.
    This computes the full Hessian matrix using finite differences,
    then diagonalizes it to get frequencies.
    """
    natoms = positions.shape[0]
    ndim = 3
    total_dof = natoms * ndim
    
    # Compute Hessian matrix using 5-point finite differences
    hessian = torch.zeros((total_dof, total_dof), dtype=torch.double)
    
    # Coefficients for 5-point formula
    coeffs = torch.tensor([1.0, -8.0, 8.0, -1.0])
    offsets = torch.tensor([-2, -1, 1, 2])
    
    print("  Computing numerical Hessian matrix...")
    for i in range(natoms):
        for j in range(ndim):
            idx1 = i * ndim + j
            print(f"    Processing atom {i+1}/{natoms}, dimension {j+1}/{ndim}")
            
            # Compute gradient at each perturbed position
            for k in range(natoms):
                for l in range(ndim):
                    idx2 = k * ndim + l
                    
                    energies = []
                    for offset, coeff in zip(offsets, coeffs):
                        # Perturb position
                        pos_perturbed = positions.clone()
                        pos_perturbed[i, j] = pos_perturbed[i, j] + offset * h
                        
                        # Create fresh calculator
                        GFN2_XTB = dxtb.GFN2_XTB
                        GFN2_XTB._per_atom_params_dict = atom_param_dict
                        
                        calc = dxtb.Calculator(numbers, GFN2_XTB, opts=opts, **dd)
                        
                        # Compute force at k,l using 5-point stencil
                        energies_inner = []
                        for offset2, coeff2 in zip(offsets, coeffs):
                            pos_double_perturbed = pos_perturbed.clone()
                            pos_double_perturbed[k, l] = pos_double_perturbed[k, l] + offset2 * h
                            
                            with torch.no_grad():
                                energy = calc.energy(pos_double_perturbed, chrg=charge, spin=spin)
                            energies_inner.append(coeff2 * energy.item())
                        
                        force_kl = -sum(energies_inner) / (12.0 * h)
                        energies.append(coeff * force_kl)
                    
                    # Second derivative (Hessian element)
                    hessian[idx1, idx2] = sum(energies) / (12.0 * h)
    
    # Symmetrize Hessian
    hessian = 0.5 * (hessian + hessian.T)
    
    # Get masses
    try:
        from tad_mctc.data import pse
        masses = torch.tensor([pse.ATOMIC_MASSES[int(n)] for n in numbers], 
                             dtype=torch.double, device=positions.device)
    except:
        # Fallback to approximate masses
        masses = torch.ones(natoms, dtype=torch.double, device=positions.device)
    
    # Mass-weight the Hessian
    mass_sqrt = torch.sqrt(masses).repeat_interleave(ndim)
    mass_weighted_hessian = hessian / torch.outer(mass_sqrt, mass_sqrt)
    
    # Diagonalize to get frequencies
    eigenvalues, _ = torch.linalg.eigh(mass_weighted_hessian)
    
    # Convert eigenvalues to frequencies
    # freq = sign(eigenvalue) * sqrt(|eigenvalue|)
    frequencies = torch.sign(eigenvalues) * torch.sqrt(torch.abs(eigenvalues))
    
    # Convert to cm^-1
    try:
        from tad_mctc.units import AU2RCM
        frequencies = frequencies * AU2RCM
    except ImportError:
        AU2RCM = 5140.487
        frequencies = frequencies * AU2RCM
    
    return frequencies


##############################################################################
# Legacy Code: Parameter Gradient Functions (Not Currently Used)
##############################################################################

def numerical_gradient_5point(numbers, param_dict, param_key, positions, charge, spin, opts, dd, h=1e-4):
    """
    Calculate numerical gradient using 5-point central difference method.
    
    Args:
        numbers: Atomic numbers
        param_dict: Dictionary of all parameters
        param_key: Key of the parameter to differentiate
        positions: Atomic positions
        charge: Molecular charge
        spin: Spin state
        opts: Calculator options
        dd: Device and dtype dictionary
        h: Step size for finite differences
    
    Returns:
        Numerical gradient tensor with same shape as param_dict[param_key]
    """
    original_value = param_dict[param_key].clone()
    shape = original_value.shape
    
    # Flatten the parameter for easier indexing
    flat_param = original_value.flatten()
    grad_flat = torch.zeros_like(flat_param)
    
    # Coefficients for 5-point formula
    coeffs = torch.tensor([1.0, -8.0, 8.0, -1.0])
    offsets = torch.tensor([-2, -1, 1, 2])
    
    # Calculate gradient for each element
    for i in range(len(flat_param)):
        energies = []
        
        for offset, coeff in zip(offsets, coeffs):
            # Create perturbed parameter
            perturbed = flat_param.clone()
            perturbed[i] = perturbed[i] + offset * h
            
            # Create a new parameter dictionary with the perturbed value
            perturbed_dict = {}
            for key, value in param_dict.items():
                if key == param_key:
                    perturbed_dict[key] = perturbed.reshape(shape).detach().requires_grad_(False)
                else:
                    perturbed_dict[key] = value.detach().clone().requires_grad_(False)
            
            # Create a fresh calculator for this evaluation
            GFN2_XTB_TEMP = dxtb.GFN2_XTB
            GFN2_XTB_TEMP._per_atom_params_dict = perturbed_dict
            
            calc_temp = dxtb.Calculator(
                numbers,
                GFN2_XTB_TEMP,
                opts=opts,
                **dd
            )
            
            # Calculate energy (disable grad to save memory)
            with torch.no_grad():
                energy = calc_temp.energy(positions, chrg=charge, spin=spin)
            energies.append(coeff * energy.item())
        
        # Apply 5-point formula
        grad_flat[i] = sum(energies) / (12.0 * h)
    
    # Restore original parameter
    param_dict[param_key] = original_value
    
    return grad_flat.reshape(shape)


def autograd_gradient(energy, param):
    """
    Calculate gradient using PyTorch autograd.
    
    Args:
        energy: Energy tensor
        param: Parameter tensor
    
    Returns:
        Gradient tensor or None if gradient is not available
    """
    if param.requires_grad:
        grad = torch.autograd.grad(
            energy, param,
            grad_outputs=torch.ones_like(energy),
            retain_graph=True,
            allow_unused=True
        )[0]
        return grad
    return None


def compare_gradients(autograd_grad, numerical_grad, param_name):
    """
    Compare autograd and numerical gradients and compute error metrics.
    
    Args:
        autograd_grad: Gradient from autograd
        numerical_grad: Gradient from numerical method
        param_name: Name of the parameter
    
    Returns:
        Dictionary containing error metrics
    """
    if autograd_grad is None:
        return {
            "param": param_name,
            "status": "No autograd gradient",
            "max_abs_error": None,
            "mean_abs_error": None,
            "rmse": None,
            "max_rel_error": None,
        }
    
    # Calculate errors
    abs_error = torch.abs(autograd_grad - numerical_grad)
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    rmse = torch.sqrt((abs_error ** 2).mean()).item()
    
    # Relative error (avoid division by zero)
    denominator = torch.maximum(torch.abs(autograd_grad), torch.abs(numerical_grad))
    denominator = torch.maximum(denominator, torch.tensor(1e-10))
    rel_error = abs_error / denominator
    max_rel_error = rel_error.max().item()
    
    return {
        "param": param_name,
        "status": "Compared",
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "max_rel_error": max_rel_error,
        "autograd_grad_norm": torch.norm(autograd_grad).item(),
        "numerical_grad_norm": torch.norm(numerical_grad).item(),
    }


def print_comparison_table(results):
    """Print formatted comparison table."""
    print("\n" + "="*120)
    print(f"{'Parameter':<20} {'Status':<20} {'Max Abs Err':<15} {'Mean Abs Err':<15} {'RMSE':<15} {'Max Rel Err':<15}")
    print("="*120)
    
    for result in results:
        param = result["param"]
        status = result["status"]
        
        if status == "Compared":
            print(f"{param:<20} {status:<20} {result['max_abs_error']:<15.3e} "
                  f"{result['mean_abs_error']:<15.3e} {result['rmse']:<15.3e} "
                  f"{result['max_rel_error']:<15.3e}")
        else:
            print(f"{param:<20} {status:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("="*120)


def main():
    """Main function to compare numerical vs autograd for Energy, Forces, and Frequencies."""
    
    # Set up device and dtype
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dd: DD = {"device": device, "dtype": torch.double}
    
    # Load molecule data
    path = Path(__file__).resolve().parent / "molecules" / "test.xyz"
    numbers, positions = read.read(path, ftype="xyz", **dd)
    
    print("="*100)
    print("NUMERICAL VS AUTOGRAD COMPARISON FOR ENERGY, FORCES, AND FREQUENCIES")
    print("="*100)
    print(f"\nMolecule: {path.name}")
    print(f"Number of atoms: {len(numbers)}")
    print(f"Device: {device}")
    print(f"Dtype: {dd['dtype']}")
    
    # Calculator options
    opts = {
        "maxiter": 100,
        "mixer": "anderson",
        "scf_mode": "full",
        "verbosity": 0,
        "f_atol": 1e-10,
        "x_atol": 1e-10,

        "per_atom": True
    }
    
    # Define atom parameter dictionary (all values set to 0)
    atom_param_dict = {
        # Global parameters (scalars)
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
        
        # Per-atom parameters (1D)
        "gam": [0.0, 0.0],
        "gam3": [0.0, 0.0],
        "zeff": [0, 0],
        "arep": [0, 0],
        "en": [0, 0],
        "dkernel": [0, 0],
        "qkernel": [0, 0],
        "mprad": [0, 0],
        "mpvcn": [0, 0],
        "rcov": [0, 0],
        "arad": [0, 0],
        
        # Per-atom shell parameters (2D)
        "levels": [[0,0,0], [0,0,0]],
        "slater": [[0.0,0.0,0.0], [0.0,0.0,0.0]],
        "shpoly": [[0,0,0], [0,0,0]],
        "kcn": [[0,0,0], [0,0,0]],
        "lgam": [[0.0,0.0,0.0], [0.0,0.0,0.0]],
        "3rd_scale": [[0, 0, 0], [0, 0, 0]],
    }
    
    # Expand parameters to match number of atoms
    natom = len(numbers)
    for key, value in atom_param_dict.items():
        if isinstance(value, list):
            atom_param_dict[key] = [value[0]] * natom
        else:
            atom_param_dict[key] = value
    
    # Convert to tensors (no requires_grad needed for this comparison)
    for key, value in atom_param_dict.items():
        if isinstance(value, list) or isinstance(value, float):
            atom_param_dict[key] = torch.tensor(value, dtype=torch.double, requires_grad=False)
        else:
            raise ValueError(f"Invalid param value type for {key}: {type(value)}")
    
    # Molecular properties
    charge = 1
    spin = 0
    
    # Step size for numerical derivatives
    h_energy = 1e-5
    h_forces = 1e-5
    h_frequencies = 1e-4  # Larger step for second derivatives
    
    ##########################################################################
    # 1. ENERGY COMPARISON
    ##########################################################################
    print("\n" + "="*100)
    print("1. ENERGY COMPARISON")
    print("="*100)
    
    # Create calculator for autograd
    print("\nComputing energy with autograd...")
    GFN2_XTB_AUTOGRAD = dxtb.GFN2_XTB
    GFN2_XTB_AUTOGRAD._per_atom_params_dict = atom_param_dict
    calc_autograd = dxtb.Calculator(numbers, GFN2_XTB_AUTOGRAD, opts=opts, **dd)
    
    start_time = time.time()
    energy_autograd = compute_energy_autograd(calc_autograd, positions, charge, spin)
    time_autograd_energy = time.time() - start_time
    
    print(f"  Energy (autograd): {energy_autograd.item():.12f} Hartree")
    print(f"  Time: {time_autograd_energy:.4f} seconds")
    
    # Compute numerical energy (should be the same, just baseline check)
    print("\nComputing energy with numerical method...")
    start_time = time.time()
    energy_numerical = compute_energy_numerical(numbers, positions, charge, spin, opts, dd, atom_param_dict, h=h_energy)
    time_numerical_energy = time.time() - start_time
    
    print(f"  Energy (numerical): {energy_numerical.item():.12f} Hartree")
    print(f"  Time: {time_numerical_energy:.4f} seconds")
    
    # Compare energies
    energy_diff = torch.abs(energy_autograd - energy_numerical).item()
    print(f"\n  Absolute difference: {energy_diff:.3e} Hartree")
    
    ##########################################################################
    # 2. FORCES COMPARISON
    ##########################################################################
    print("\n" + "="*100)
    print("2. FORCES COMPARISON")
    print("="*100)
    
    # Compute forces with autograd
    print("\nComputing forces with autograd...")
    GFN2_XTB_AUTOGRAD = dxtb.GFN2_XTB
    GFN2_XTB_AUTOGRAD._per_atom_params_dict = atom_param_dict
    calc_autograd = dxtb.Calculator(numbers, GFN2_XTB_AUTOGRAD, opts=opts, **dd)
    
    start_time = time.time()
    forces_autograd = compute_forces_autograd(calc_autograd, positions, charge, spin)
    time_autograd_forces = time.time() - start_time
    
    print(f"  Forces computed (shape: {forces_autograd.shape})")
    print(f"  Max force magnitude: {torch.abs(forces_autograd).max().item():.6e} Hartree/Bohr")
    print(f"  RMS force: {torch.sqrt((forces_autograd**2).mean()).item():.6e} Hartree/Bohr")
    print(f"  Time: {time_autograd_forces:.4f} seconds")
    
    # Compute forces with numerical method
    print("\nComputing forces with numerical method (5-point)...")
    start_time = time.time()
    forces_numerical = compute_forces_numerical(numbers, positions, charge, spin, opts, dd, atom_param_dict, h=h_forces)
    time_numerical_forces = time.time() - start_time
    
    print(f"  Forces computed (shape: {forces_numerical.shape})")
    print(f"  Max force magnitude: {torch.abs(forces_numerical).max().item():.6e} Hartree/Bohr")
    print(f"  RMS force: {torch.sqrt((forces_numerical**2).mean()).item():.6e} Hartree/Bohr")
    print(f"  Time: {time_numerical_forces:.4f} seconds")
    
    # Compare forces
    forces_diff = forces_autograd - forces_numerical
    max_abs_error = torch.abs(forces_diff).max().item()
    mean_abs_error = torch.abs(forces_diff).mean().item()
    rmse = torch.sqrt((forces_diff**2).mean()).item()
    
    # Relative error
    denominator = torch.maximum(torch.abs(forces_autograd), torch.abs(forces_numerical))
    denominator = torch.maximum(denominator, torch.tensor(1e-10))
    rel_error = torch.abs(forces_diff) / denominator
    max_rel_error = rel_error.max().item()
    
    print("\n  FORCES ERROR ANALYSIS:")
    print(f"    Max absolute error: {max_abs_error:.3e} Hartree/Bohr")
    print(f"    Mean absolute error: {mean_abs_error:.3e} Hartree/Bohr")
    print(f"    RMSE: {rmse:.3e} Hartree/Bohr")
    print(f"    Max relative error: {max_rel_error:.3e}")
    
    ##########################################################################
    # 3. FREQUENCIES COMPARISON
    ##########################################################################
    print("\n" + "="*100)
    print("3. VIBRATIONAL FREQUENCIES COMPARISON")
    print("="*100)
    
    # Compute frequencies with autograd (using the provided function)
    print("\nComputing frequencies with autograd (analytical Hessian)...")
    GFN2_XTB_AUTOGRAD = dxtb.GFN2_XTB
    GFN2_XTB_AUTOGRAD._per_atom_params_dict = atom_param_dict
    calc_autograd = dxtb.Calculator(numbers, GFN2_XTB_AUTOGRAD, opts=opts, **dd)
    
    start_time = time.time()
    pos_freq = positions.clone().requires_grad_(True)
    freqs_autograd = compute_frequencies_dxtb(calc_autograd, pos_freq, charge=charge, 
                                              return_modes=False, convert_units=True, cut_freqs=False)
    time_autograd_freqs = time.time() - start_time
    
    print(f"  Frequencies computed: {len(freqs_autograd)} modes")
    print(f"  Time: {time_autograd_freqs:.4f} seconds")
    print(f"\n  Frequencies (cm⁻¹):")
    for i, freq in enumerate(freqs_autograd):
        print(f"    Mode {i+1:3d}: {freq.item():12.4f} cm⁻¹")
    
    # Compute frequencies with numerical method
    print("\nComputing frequencies with numerical method (5-point Hessian)...")
    print("  WARNING: This will be very slow (computing full Hessian)...")
    start_time = time.time()
    freqs_numerical = compute_frequencies_numerical(numbers, positions, charge, spin, opts, dd, atom_param_dict, h=h_frequencies)
    time_numerical_freqs = time.time() - start_time
    
    print(f"  Frequencies computed: {len(freqs_numerical)} modes")
    print(f"  Time: {time_numerical_freqs:.4f} seconds")
    print(f"\n  Frequencies (cm⁻¹):")
    for i, freq in enumerate(freqs_numerical):
        print(f"    Mode {i+1:3d}: {freq.item():12.4f} cm⁻¹")
    
    # Compare frequencies
    freqs_diff = freqs_autograd - freqs_numerical
    max_abs_error_freq = torch.abs(freqs_diff).max().item()
    mean_abs_error_freq = torch.abs(freqs_diff).mean().item()
    rmse_freq = torch.sqrt((freqs_diff**2).mean()).item()
    
    # Relative error
    denominator_freq = torch.maximum(torch.abs(freqs_autograd), torch.abs(freqs_numerical))
    denominator_freq = torch.maximum(denominator_freq, torch.tensor(1e-10))
    rel_error_freq = torch.abs(freqs_diff) / denominator_freq
    max_rel_error_freq = rel_error_freq.max().item()
    
    print("\n  FREQUENCIES ERROR ANALYSIS:")
    print(f"    Max absolute error: {max_abs_error_freq:.3e} cm⁻¹")
    print(f"    Mean absolute error: {mean_abs_error_freq:.3e} cm⁻¹")
    print(f"    RMSE: {rmse_freq:.3e} cm⁻¹")
    print(f"    Max relative error: {max_rel_error_freq:.3e}")
    
    ##########################################################################
    # SUMMARY
    ##########################################################################
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    print("\nCOMPUTATION TIMES:")
    print(f"  Energy (autograd):        {time_autograd_energy:.4f} s")
    print(f"  Energy (numerical):       {time_numerical_energy:.4f} s")
    print(f"  Forces (autograd):        {time_autograd_forces:.4f} s")
    print(f"  Forces (numerical):       {time_numerical_forces:.4f} s")
    print(f"  Frequencies (autograd):   {time_autograd_freqs:.4f} s")
    print(f"  Frequencies (numerical):  {time_numerical_freqs:.4f} s")
    
    print("\nERROR SUMMARY:")
    print(f"  Energy:")
    print(f"    Absolute difference:    {energy_diff:.3e} Hartree")
    print(f"  Forces:")
    print(f"    Max absolute error:     {max_abs_error:.3e} Hartree/Bohr")
    print(f"    Max relative error:     {max_rel_error:.3e}")
    print(f"  Frequencies:")
    print(f"    Max absolute error:     {max_abs_error_freq:.3e} cm⁻¹")
    print(f"    Max relative error:     {max_rel_error_freq:.3e}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()

