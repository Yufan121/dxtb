#!/usr/bin/env python3
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
Geometry optimization with GFN2-xTB and per-atom parameters.

This example combines:
1. Parameter initialization from params_dict.json (forces_gfn2_dipole.py style)
2. Geometry optimization using xitorch.optimize.minimize (issues/187/run.py style)
"""
from pathlib import Path
import json
import time
import torch
from tad_mctc.io import read

import dxtb
from dxtb.typing import DD
from dxtb._src.components.interactions import new_efield

############################################
# Setup
############################################

# Set up device and dtype
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dd: DD = {"device": device, "dtype": torch.double}

print(f"Using device: {device}")
print(f"Using dtype: {dd['dtype']}\n")

# Load molecule data from XYZ file
path = Path(__file__).resolve().parent / "molecules" / "FH-BH2.xyz"
numbers, positions = read.read(path, ftype="xyz", **dd)

print(f"Molecule loaded from: {path}")
print(f"Number of atoms: {len(numbers)}")
print(f"Atomic numbers: {numbers}")
print(f"Initial positions:\n{positions}\n")

# Calculator options
opts = {
    "verbosity": 0,
    "per_atom": True  # Enable per-atom parameters
}

############################################
# Parameter Initialization
############################################

print("Loading per-atom parameters from params_dict.json...")

# Load parameters from JSON file
params_file = Path(__file__).resolve().parent / "params_dict.json"
with open(params_file, "r") as f:
    params_dict = json.load(f)
    atom_param_dict = params_dict["params_dict"]

print(f"Parameters loaded from: {params_file}")

# Convert all parameter values to tensors with requires_grad=True
for key, value in atom_param_dict.items():
    if isinstance(value, list) or isinstance(value, float):
        atom_param_dict[key] = torch.tensor(value, dtype=torch.double, requires_grad=True)
    else:
        raise ValueError(f"Invalid param value type for {key}: {type(value)}")

print(f"Number of parameter keys: {len(atom_param_dict)}\n")

# Set up GFN2-xTB with per-atom parameters
GFN2_XTB_ATOM = dxtb.GFN2_XTB
GFN2_XTB_ATOM._per_atom_params_dict = atom_param_dict

# Set up electric field (optional, set to zero)
efield_vec = torch.zeros(3, dtype=torch.double, device=device, requires_grad=False)
efield = new_efield(efield_vec)

# Initialize calculator with GFN2-xTB and per-atom parameters
print("Initializing Calculator with GFN2-xTB and per-atom parameters...")
calc = dxtb.Calculator(
    numbers,
    GFN2_XTB_ATOM,
    interaction=[efield],
    opts=opts,
    **dd
)
print("Calculator initialized.\n")

############################################
# Geometry Optimization
############################################

def main() -> int:
    """Main optimization routine."""
    
    assert calc.integrals.hcore is not None
    
    from dxtb._src.exlibs.xitorch.optimize import minimize
    
    print("=" * 60)
    print("Starting Geometry Optimization")
    print("=" * 60)
    
    # Set up energy function for optimization
    def get_energy(pos: torch.Tensor) -> torch.Tensor:
        """
        Energy function for optimization.
        
        Parameters
        ----------
        pos : torch.Tensor
            Atomic positions (shape: (nat, 3))
            
        Returns
        -------
        torch.Tensor
            Total energy (scalar)
        """
        return calc.get_energy(pos)
    
    # Initial energy
    initial_energy = get_energy(positions)
    print(f"\nInitial energy: {initial_energy.item():.10f} Hartree\n")
    
    # Run optimization
    time_start = time.time()
    
    print("Running optimization (method=gd, maxiter=200, step=1e-2)...")
    print("-" * 60)
    
    optimized_positions = minimize(
        get_energy,
        positions,
        method="gd",      # Gradient descent
        maxiter=200,      # Maximum iterations
        step=1e-2,        # Step size
        verbose=True,     # Print progress
    )
    
    time_end = time.time()
    
    print("-" * 60)
    print(f"Optimization completed in {time_end - time_start:.2f} seconds\n")
    
    # Final energy
    final_energy = get_energy(optimized_positions)
    energy_change = final_energy.item() - initial_energy.item()
    
    print("=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Initial energy:    {initial_energy.item():.10f} Hartree")
    print(f"Final energy:      {final_energy.item():.10f} Hartree")
    print(f"Energy change:     {energy_change:.10f} Hartree")
    print(f"Energy lowered by: {abs(energy_change) * 27.2114:.6f} eV\n")
    
    print("Initial geometry (Angstrom):")
    print(positions.detach().cpu().numpy())
    print()
    
    print("Optimized geometry (Angstrom):")
    print(optimized_positions.detach().cpu().numpy())
    print()

    # print RMSD between initial and optimized geometry
    rmsd = torch.norm(optimized_positions - positions, dim=1).mean().item()
    print(f"RMSD between initial and optimized geometry: {rmsd:.6f} Angstrom")
    print()
    
    # Calculate displacement
    displacement = torch.norm(optimized_positions - positions, dim=1)
    max_displacement = displacement.max().item()
    avg_displacement = displacement.mean().item()
    
    print(f"Maximum atomic displacement: {max_displacement:.6f} Angstrom")
    print(f"Average atomic displacement: {avg_displacement:.6f} Angstrom")
    print()
    
    # Save optimized geometry to file (optional)
    output_file = Path(__file__).resolve().parent / "optimized_geometry.xyz"
    with open(output_file, "w") as f:
        f.write(f"{len(numbers)}\n")
        f.write(f"Optimized geometry, Energy = {final_energy.item():.10f} Hartree\n")
        for i, (num, pos) in enumerate(zip(numbers, optimized_positions.detach().cpu())):
            # Convert atomic number to element symbol (simple mapping)
            element_map = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F"}
            element = element_map.get(num.item(), "X")
            f.write(f"{element:2s} {pos[0]:15.10f} {pos[1]:15.10f} {pos[2]:15.10f}\n")
    
    print(f"Optimized geometry saved to: {output_file}")
    print()
    
    # Check if energy is NaN
    if torch.isnan(final_energy).any():
        print("WARNING: Final energy is NaN!")
        print(f"atom_param_dict: {atom_param_dict}")
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
