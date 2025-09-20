#!/usr/bin/env python3
"""
Script to compare GFN2-xTB calculations with reference data.

This script reads reference coordinates and forces from two files:
- output.xyz: Contains molecular coordinates in standard XYZ format
- output_force.xyz: Contains reference forces in eV/Å and energy in Hartree

It then runs GFN2-xTB calculations using dxtb and compares the results.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Tuple, Dict, Any
import re
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import os

# Add dxtb to path if needed
sys.path.insert(0, str(Path(__file__).parent))

import dxtb
from dxtb.typing import DD
from tad_mctc.io import read


def read_reference_data(coords_file: str, forces_file: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, int]:
    """
    Read reference coordinates, forces, energy, charge, and multiplicity from files.
    
    Args:
        coords_file: Path to XYZ file with coordinates
        forces_file: Path to file with reference forces
        
    Returns:
        Tuple of (numbers, positions, forces, energy, charge, multiplicity)
    """
    # Read coordinates from XYZ file
    coords_path = Path(coords_file)
    if not coords_path.exists():
        raise FileNotFoundError(f"Coordinates file not found: {coords_file}")
    
    # Read using tad_mctc
    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    numbers, positions = read.read(coords_path, ftype="xyz", **dd)
    
    # convert positions from Angstron to Bohr
    positions = positions * 1.8897259886
    
    # Read forces and energy from forces file
    forces_path = Path(forces_file)
    if not forces_path.exists():
        raise FileNotFoundError(f"Forces file not found: {forces_file}")
    
    with open(forces_path, 'r') as f:
        lines = f.readlines()
    
    # Extract energy from header (in Hartree)
    energy_line = None
    charge_line = None
    multiplicity_line = None
    
    for line in lines:
        if line.startswith("# Energy:"):
            energy_line = line
        elif line.startswith("# Charge:"):
            charge_line = line
        elif line.startswith("# Multiplicity:"):
            multiplicity_line = line
    
    if not energy_line:
        raise ValueError("Energy not found in forces file header")
    
    # Parse energy (in Hartree)
    energy_match = re.search(r"Energy:\s+([+-]?\d+\.?\d*)", energy_line)
    if not energy_match:
        raise ValueError("Could not parse energy from forces file")
    energy = float(energy_match.group(1))
    
    # Parse charge
    charge = 0
    if charge_line:
        charge_match = re.search(r"Charge:\s+([+-]?\d+\.?\d*)", charge_line)
        if charge_match:
            charge = int(round(float(charge_match.group(1))))
    
    # Parse multiplicity
    multiplicity = 1
    if multiplicity_line:
        mult_match = re.search(r"Multiplicity:\s+(\d+)", multiplicity_line)
        if mult_match:
            multiplicity = int(mult_match.group(1))
    
    # Read forces (skip header lines and first data line which is atom count)
    force_lines = []
    in_data = False
    for line in lines:
        if line.strip().startswith("#"):
            continue
        if not in_data and line.strip().isdigit():
            in_data = True
            continue
        if in_data and line.strip():
            force_lines.append(line.strip())
    
    # Parse forces (in eV/Å)
    forces = []
    for line in force_lines:
        parts = line.split()
        if len(parts) >= 5:  # atom_index element fx fy fz
            fx, fy, fz = float(parts[2]), float(parts[3]), float(parts[4])
            forces.append([fx, fy, fz])
    
    forces = torch.tensor(forces, dtype=torch.double)
    
    return numbers, positions, forces, energy, charge, multiplicity


def convert_units_energy(eV: float) -> float:
    """Convert energy from eV to Hartree."""
    return eV / 27.211386245988  # eV to Hartree


def convert_units_force(eV_per_angstrom: float) -> float:
    """Convert force from eV/Å to Hartree/Bohr."""
    # eV/Å to Hartree/Bohr
    # 1 eV = 1/27.211386245988 Hartree
    # 1 Å = 1.8897259886 Bohr
    return eV_per_angstrom / (27.211386245988 * 1.8897259886)


def convert_units_force_tensor(forces_eV_per_angstrom: torch.Tensor) -> torch.Tensor:
    """Convert forces tensor from eV/Å to Hartree/Bohr."""
    conversion_factor = 1.0 / (27.211386245988 * 1.8897259886)
    return forces_eV_per_angstrom * conversion_factor


def run_gfn2_calculation(numbers: torch.Tensor, positions: torch.Tensor, 
                        charge: int, multiplicity: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run GFN2-xTB calculation using dxtb with zero parameters setup.
    
    Args:
        numbers: Atomic numbers
        positions: Atomic positions
        charge: Molecular charge
        multiplicity: Spin multiplicity
        device: Device to run calculation on
        
    Returns:
        Tuple of (energy, forces) in Hartree and Hartree/Bohr respectively
    """
    # Set up device and dtype
    device_obj = torch.device(device)
    dd: DD = {"device": device_obj, "dtype": torch.double}
    
    # Set up zero parameters with correct shapes (from forces_gfn2_big.py)
    natom = len(numbers)
    
    atom_param_dict = {
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

        ###### atoms' parameters
        "levels": [[0,0,0], [0,0,0]],   
        "slater": [[0.0,0.0,0.0], [0.0,0.0,0.0]],   
        "shpoly": [[0,0,0], [0,0,0]],   
        "kcn": [[0,0,0], [0,0,0]],
        "gam": [0.0, 0.0],
        "lgam": [[0.0,0.0,0.0], [0.0,0.0,0.0]],
        "gam3": [0.0, 0.0],
        "zeff": [0, 0],
        "arep": [0, 0],
        "en": [0, 0],
        # multipole parameters
        "dkernel": [0, 0],
        "qkernel": [0, 0],
        "mprad": [0, 0],
        "mpvcn": [0, 0],
        "3rd_scale": [[0, 0, 0], [0, 0, 0]],
        "rcov": [0, 0],
        "arad": [0, 0],
    }
    
    # Copy each value in atom_param_dict to match numbers
    for key, value in atom_param_dict.items():
        if isinstance(value, list):
            atom_param_dict[key] = [value[0]] * natom
        else:
            atom_param_dict[key] = value

    # Make all values tensors and requires_grad = True
    for key, value in atom_param_dict.items():
        if isinstance(value, list) or isinstance(value, float):
            atom_param_dict[key] = torch.tensor(value, dtype=torch.double, requires_grad=True)
        else:
            raise ValueError(f"Invalid param value type for {key}: {type(value)}")
    
    # Calculator options
    opts = {
        "verbosity": 0,
        "per_atom": True  # Use per-atom parameters
    }
    
    # Set up GFN2_XTB_ATOM with the parameter dictionary
    GFN2_XTB_ATOM = dxtb.GFN2_XTB
    # Set it to Param class's vars for access
    GFN2_XTB_ATOM._per_atom_params_dict = atom_param_dict  # type: ignore
    
    # Initialize calculator
    calc = dxtb.Calculator(
        numbers,
        GFN2_XTB_ATOM,
        opts=opts,
        **dd
    )
    
    # charge = -1
    # multiplicity = 0
    
    # Calculate energy
    pos = positions.clone().requires_grad_(True)
    energy = calc.energy(pos, chrg=charge, spin=multiplicity)
    
    # Calculate forces as negative gradient of energy
    (g,) = torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy), 
                              retain_graph=True, create_graph=True)
    forces = -g
    
    return energy, forces


def calculate_rmse(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """Calculate Root Mean Square Error between predicted and reference values."""
    mse = torch.mean((predicted - reference) ** 2)
    return torch.sqrt(mse).item()


def worker_calculate_multiplicity(args):
    """
    Worker function for multiprocessing to calculate a single multiplicity.
    
    Args:
        args: Tuple of (numbers, positions, charge, multiplicity, ref_energy, ref_forces_converted)
        
    Returns:
        Dictionary with results for this multiplicity
    """
    numbers, positions, charge, multiplicity, ref_energy, ref_forces_converted = args
    
    # Set process to use CPU only and single thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    try:
        # Run calculation
        calc_energy, calc_forces = run_gfn2_calculation(
            numbers, positions, charge, multiplicity, "cpu"
        )
        
        # Check for NaN values
        if torch.isnan(calc_energy).any():
            return {
                'multiplicity': multiplicity,
                'success': False,
                'error': 'Energy is NaN'
            }
            
        if torch.isnan(calc_forces).any().any():
            return {
                'multiplicity': multiplicity,
                'success': False,
                'error': 'Forces contain NaN'
            }
        
        # Calculate errors
        energy_error = abs(calc_energy.item() - ref_energy)
        energy_error_meV = energy_error * 27.211386245988 * 1000
        
        force_rmse = calculate_rmse(calc_forces, ref_forces_converted)
        force_rmse_eV_per_angstrom = force_rmse * (27.211386245988 * 1.8897259886)
        
        return {
            'multiplicity': multiplicity,
            'energy': calc_energy.item(),
            'forces': calc_forces.detach(),  # Detach to remove grad_fn for serialization
            'energy_error': energy_error,
            'energy_error_meV': energy_error_meV,
            'force_rmse': force_rmse,
            'force_rmse_eV_per_angstrom': force_rmse_eV_per_angstrom,
            'success': True
        }
        
    except Exception as e:
        return {
            'multiplicity': multiplicity,
            'success': False,
            'error': str(e)
        }


def run_multiple_multiplicities(numbers: torch.Tensor, positions: torch.Tensor, 
                               charge: int, ref_energy: float, ref_forces_converted: torch.Tensor,
                               device: str = "cpu", max_multiplicity: int = 6) -> Dict[str, Any]:
    """
    Run GFN2-xTB calculation with multiple multiplicities using multiprocessing and find the minimum energy.
    
    Args:
        numbers: Atomic numbers
        positions: Atomic positions
        charge: Molecular charge
        ref_energy: Reference energy in Hartree
        ref_forces_converted: Reference forces in Hartree/Bohr
        device: Device to run calculation on (ignored, always uses CPU for multiprocessing)
        max_multiplicity: Maximum multiplicity to test
        
    Returns:
        Dictionary with results for all multiplicities and the best one
    """
    print(f"\nRunning calculations for multiplicities 0 to {max_multiplicity} using multiprocessing...")
    print("=" * 80)
    
    # Determine number of processes (use all available cores)
    num_processes = min(max_multiplicity + 1, cpu_count())
    print(f"Using {num_processes} processes (one per core)")
    
    # Prepare arguments for each process
    args_list = []
    for mult in range(max_multiplicity + 1):
        args_list.append((numbers, positions, charge, mult, ref_energy, ref_forces_converted))
    
    # Run calculations in parallel
    print("Starting parallel calculations...")
    with Pool(processes=num_processes) as pool:
        results_list = pool.map(worker_calculate_multiplicity, args_list)
    
    # Process results
    results = {}
    energies = []
    multiplicities = []
    
    print("\nProcessing results...")
    for result in results_list:
        mult = result['multiplicity']
        results[mult] = result
        
        if result['success']:
            energies.append(result['energy'])
            multiplicities.append(mult)
            print(f"  ✓ Multiplicity {mult}: Energy = {result['energy']:.6f} Hartree, "
                  f"Error = {result['energy_error_meV']:.2f} meV, "
                  f"Force RMSE = {result['force_rmse_eV_per_angstrom']:.6f} eV/Å")
        else:
            print(f"  ❌ Multiplicity {mult}: FAILED - {result['error']}")
    
    if not energies:
        raise ValueError("No successful calculations found!")
    
    # Find minimum energy
    min_energy_idx = np.argmin(energies)
    best_multiplicity = multiplicities[min_energy_idx]
    best_energy = energies[min_energy_idx]
    
    print(f"\n" + "=" * 80)
    print("SUMMARY OF ALL MULTIPLICITIES")
    print("=" * 80)
    
    for mult in sorted(results.keys()):
        if results[mult]['success']:
            res = results[mult]
            marker = " 🏆" if mult == best_multiplicity else ""
            print(f"Multiplicity {mult:2d}: Energy = {res['energy']:12.6f} Hartree, "
                  f"Error = {res['energy_error_meV']:8.2f} meV, "
                  f"Force RMSE = {res['force_rmse_eV_per_angstrom']:8.6f} eV/Å{marker}")
        else:
            print(f"Multiplicity {mult:2d}: FAILED - {results[mult]['error']}")
    
    print(f"\n🏆 BEST RESULT: Multiplicity {best_multiplicity}")
    print(f"   Energy: {best_energy:.6f} Hartree")
    print(f"   Energy error: {results[best_multiplicity]['energy_error_meV']:.2f} meV")
    print(f"   Force RMSE: {results[best_multiplicity]['force_rmse_eV_per_angstrom']:.6f} eV/Å")
    
    return {
        'all_results': results,
        'best_multiplicity': best_multiplicity,
        'best_energy': best_energy,
        'best_result': results[best_multiplicity]
    }


def main():
    """Main function to run the comparison."""
    # File paths
    coords_file = "/scratch/kx58/yx7184/githubs/NNxTB/nnxtb_autograd/data_processing/output.xyz"
    forces_file = "/scratch/kx58/yx7184/githubs/NNxTB/nnxtb_autograd/data_processing/output_force.xyz"
    
    print("=" * 80)
    print("GFN2-xTB vs Reference Data Comparison")
    print("=" * 80)
    
    try:
        # Read reference data
        print("Reading reference data...")
        numbers, positions, ref_forces, ref_energy, charge, multiplicity = read_reference_data(
            coords_file, forces_file
        )
        
        print(f"Molecular system:")
        print(f"  Number of atoms: {len(numbers)}")
        print(f"  Charge: {charge}")
        print(f"  Multiplicity: {multiplicity}")
        print(f"  Reference energy: {ref_energy:.6f} Hartree")
        print(f"  Reference forces shape: {ref_forces.shape}")
        
        # Convert reference forces from eV/Å to Hartree/Bohr
        ref_forces_converted = convert_units_force_tensor(ref_forces)
        
        # Run GFN2-xTB calculation with multiple multiplicities
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Run calculations for all multiplicities
        mult_results = run_multiple_multiplicities(
            numbers, positions, charge, ref_energy, ref_forces_converted, device, max_multiplicity=6
        )
        
        # Get the best result
        best_result = mult_results['best_result']
        calc_energy = torch.tensor(best_result['energy'])
        calc_forces = best_result['forces']
        best_multiplicity = mult_results['best_multiplicity']
        
        # Detailed analysis of the best result
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS OF BEST RESULT")
        print("=" * 80)
        
        # Energy comparison
        energy_error = best_result['energy_error']
        energy_error_meV = best_result['energy_error_meV']
        print(f"Energy comparison (Multiplicity {best_multiplicity}):")
        print(f"  Reference:  {ref_energy:.6f} Hartree")
        print(f"  Calculated: {calc_energy.item():.6f} Hartree")
        print(f"  Error:      {energy_error:.6f} Hartree ({energy_error_meV:.2f} meV)")
        
        # Force comparison
        force_rmse = best_result['force_rmse']
        force_rmse_eV_per_angstrom = best_result['force_rmse_eV_per_angstrom']
        
        print(f"\nForce comparison:")
        print(f"  RMSE: {force_rmse:.6f} Hartree/Bohr ({force_rmse_eV_per_angstrom:.6f} eV/Å)")
        
        # Per-atom force statistics
        force_errors = torch.abs(calc_forces - ref_forces_converted)
        max_force_error = torch.max(force_errors).item()
        mean_force_error = torch.mean(force_errors).item()
        
        print(f"  Max force error:  {max_force_error:.6f} Hartree/Bohr")
        print(f"  Mean force error: {mean_force_error:.6f} Hartree/Bohr")
        
        # Component-wise force errors
        print(f"\nForce component errors (Hartree/Bohr):")
        for i, comp in enumerate(['x', 'y', 'z']):
            comp_rmse = calculate_rmse(calc_forces[:, i], ref_forces_converted[:, i])
            print(f"  {comp}-component RMSE: {comp_rmse:.6f}")
        
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Best multiplicity: {best_multiplicity}")
        print(f"Energy error: {energy_error_meV:.2f} meV")
        print(f"Force RMSE:  {force_rmse_eV_per_angstrom:.6f} eV/Å")
        
        # Overall assessment
        if energy_error_meV < 1.0 and force_rmse_eV_per_angstrom < 0.1:
            print("✓ Excellent agreement with reference data")
        elif energy_error_meV < 10.0 and force_rmse_eV_per_angstrom < 1.0:
            print("✓ Good agreement with reference data")
        else:
            print("⚠ Significant differences from reference data")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
