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
Geometry optimization with GFN2-xTB and per-atom parameters using ASE BFGS.

This example combines:
1. Parameter initialization from params_dict.json (forces_gfn2_dipole.py style)
2. Geometry optimization using ASE's BFGS optimizer (issues/206/run.py style)
3. ASE Calculator wrapper for dxtb with per-atom parameters
"""
from pathlib import Path
import json
import time
import torch
import numpy as np
from tad_mctc.convert import numpy_to_tensor, tensor_to_numpy
from tad_mctc.units import AA2AU, AU2AA

import dxtb
from dxtb.typing import DD
from dxtb._src.components.interactions import new_efield

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator as AseCalculator
    from ase.optimize import BFGS
    from ase.io import write as ase_write
except ImportError:
    raise SystemExit("Please install ASE to run this example: pip install ase")


############################################
# ASE Calculator Wrapper for dxtb
############################################

class DxtbAseCalculator(AseCalculator):
    """
    ASE Calculator wrapper for dxtb with per-atom parameters.
    
    This calculator wraps the dxtb Calculator and provides an ASE-compatible
    interface for geometry optimization.
    """
    
    implemented_properties = ["energy", "forces"]
    
    def __init__(
        self,
        numbers: torch.Tensor,
        parametrization,
        atom_param_dict: dict,
        efield_vec: torch.Tensor,
        opts: dict,
        dd: DD,
        **kwargs,
    ):
        """
        Initialize the ASE calculator wrapper.
        
        Parameters
        ----------
        numbers : torch.Tensor
            Atomic numbers
        parametrization : dxtb.Param
            GFN2-xTB parametrization with per-atom parameters
        atom_param_dict : dict
            Dictionary of per-atom parameters
        efield_vec : torch.Tensor
            Electric field vector
        opts : dict
            Calculator options
        dd : DD
            Device and dtype dictionary
        """
        super().__init__(**kwargs)
        self.numbers = numbers
        self.parametrization = parametrization
        self.atom_param_dict = atom_param_dict
        self.efield_vec = efield_vec
        self.opts = opts
        self.dd = dd
        
        # Create electric field interaction
        self.efield = new_efield(efield_vec)
        
        # Initialize dxtb calculator (will be created fresh each time to avoid caching issues)
        self.dxtb_calc = None
        
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=["energy", "forces"],
        system_changes=None,
    ):
        """
        Calculate energy and forces for the given atoms.
        
        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object
        properties : list
            List of properties to calculate
        system_changes : list
            List of system changes (not used)
        """
        AseCalculator.calculate(self, atoms, properties, system_changes)
        assert atoms is not None
        
        # Convert ASE atoms to torch tensors
        # Note: ASE uses Angstrom, dxtb uses Bohr (atomic units)
        positions_ase = atoms.get_positions()  # in Angstrom
        positions = numpy_to_tensor(positions_ase * AA2AU, **self.dd)  # Convert to Bohr
        
        # Get charge from atoms.info (default to 0)
        chrg = atoms.info.get("charge", 0)
        spin = atoms.info.get("spin", 0)
        
        # Create fresh dxtb calculator with per-atom parameters
        # Note: We create a new calculator each time to avoid caching issues
        self.dxtb_calc = dxtb.Calculator(
            numbers=self.numbers,
            par=self.parametrization,
            interaction=[self.efield],
            opts=self.opts,
            **self.dd,
        )
        
        # Enable gradient computation for forces
        positions.requires_grad_(True)
        
        # Calculate forces first (this also computes energy internally)
        forces = self.dxtb_calc.get_forces(positions, chrg=chrg, spin=spin)
        
        # With cache enabled, energy is already computed
        energy = self.dxtb_calc.get_energy(positions, chrg=chrg, spin=spin)
        
        # Convert back to numpy and ASE units
        # Energy: Hartree (already in correct units)
        # Forces: Hartree/Bohr -> eV/Angstrom (ASE default)
        # Conversion: 1 Hartree/Bohr = 27.2114 eV/Angstrom * (Bohr/Angstrom) = 51.4220 eV/Angstrom
        energy_np = tensor_to_numpy(energy.sum())  # Sum over batch if needed
        forces_np = tensor_to_numpy(forces)  # in Hartree/Bohr
        
        # ASE expects forces in eV/Angstrom by default, but dxtb returns Hartree/Bohr
        # Conversion factor: 1 Hartree/Bohr = 51.42208619083232 eV/Angstrom
        HARTREE_BOHR_TO_EV_ANG = 51.42208619083232
        forces_ase = forces_np * HARTREE_BOHR_TO_EV_ANG
        
        # ASE expects energy in eV by default, but dxtb returns Hartree
        # Conversion factor: 1 Hartree = 27.211386245988 eV
        HARTREE_TO_EV = 27.211386245988
        energy_ase = energy_np * HARTREE_TO_EV
        
        # Store results
        self.results.update(
            {
                "energy": energy_ase,
                "forces": forces_ase,
            }
        )


############################################
# Setup
############################################

def main() -> int:
    """Main optimization routine using ASE BFGS."""
    
    # Set up device and dtype
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dd: DD = {"device": device, "dtype": torch.double}
    
    print("=" * 70)
    print("Geometry Optimization with ASE BFGS and dxtb GFN2-xTB")
    print("=" * 70)
    print(f"Using device: {device}")
    print(f"Using dtype: {dd['dtype']}\n")
    
    # Load molecule data from XYZ file
    from tad_mctc.io import read
    
    path = Path(__file__).resolve().parent / "molecules" / "FH-BH2.xyz"
    numbers, positions = read.read(path, ftype="xyz", **dd)
    
    print(f"Molecule loaded from: {path}")
    print(f"Number of atoms: {len(numbers)}")
    print(f"Atomic numbers: {numbers.cpu().numpy()}")
    print(f"Initial positions (Bohr):\n{positions.cpu().numpy()}\n")
    
    # Calculator options
    opts = {
        "verbosity": 0,
        "per_atom": True,  # Enable per-atom parameters
        "cache_enabled": True,  # Enable caching for efficiency
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
    
    ############################################
    # Create ASE Atoms Object
    ############################################
    
    # Convert positions from Bohr to Angstrom for ASE
    positions_angstrom = positions.detach().cpu().numpy() * AU2AA
    numbers_np = numbers.cpu().numpy()
    
    # Create ASE Atoms object
    atoms = Atoms(
        numbers=numbers_np,
        positions=positions_angstrom,
    )
    
    # Set charge and spin in atoms.info
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    
    print("ASE Atoms object created:")
    print(f"  Chemical formula: {atoms.get_chemical_formula()}")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Initial positions (Angstrom):")
    print(f"{atoms.get_positions()}\n")
    
    ############################################
    # Set up ASE Calculator
    ############################################
    
    print("Setting up ASE calculator wrapper for dxtb...")
    atoms.calc = DxtbAseCalculator(
        numbers=numbers,
        parametrization=GFN2_XTB_ATOM,
        atom_param_dict=atom_param_dict,
        efield_vec=efield_vec,
        opts=opts,
        dd=dd,
    )
    print("ASE calculator initialized.\n")
    
    # Calculate initial energy
    initial_energy = atoms.get_potential_energy()  # in eV (ASE default)
    print(f"Initial energy: {initial_energy:.10f} eV")
    print(f"               ({initial_energy / 27.211386245988:.10f} Hartree)\n")
    
    ############################################
    # Run BFGS Optimization
    ############################################
    
    print("=" * 70)
    print("Starting BFGS Optimization")
    print("=" * 70)
    print("Convergence criterion: fmax = 0.05 eV/Angstrom")
    print("Maximum steps: 200\n")
    
    # Set up BFGS optimizer
    trajectory_file = Path(__file__).resolve().parent / "optimization_trajectory.traj"
    logfile = Path(__file__).resolve().parent / "optimization.log"
    
    time_start = time.time()
    
    optimizer = BFGS(
        atoms,
        trajectory=str(trajectory_file),
        logfile=str(logfile),
    )
    
    # Run optimization
    print("Running BFGS optimization...")
    print("-" * 70)
    
    try:
        optimizer.run(fmax=0.05, steps=200)
        optimization_success = True
    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
        optimization_success = False
    
    time_end = time.time()
    
    print("-" * 70)
    print(f"Optimization completed in {time_end - time_start:.2f} seconds\n")
    
    ############################################
    # Results
    ############################################
    
    # Calculate final energy
    final_energy = atoms.get_potential_energy()  # in eV
    energy_change = final_energy - initial_energy
    
    print("=" * 70)
    print("Optimization Results")
    print("=" * 70)
    print(f"Optimization successful: {optimization_success}")
    print(f"Number of steps: {optimizer.nsteps}")
    print(f"\nInitial energy:    {initial_energy:.10f} eV")
    print(f"                   ({initial_energy / 27.211386245988:.10f} Hartree)")
    print(f"Final energy:      {final_energy:.10f} eV")
    print(f"                   ({final_energy / 27.211386245988:.10f} Hartree)")
    print(f"Energy change:     {energy_change:.10f} eV")
    print(f"                   ({energy_change / 27.211386245988:.10f} Hartree)")
    print(f"Energy lowered by: {abs(energy_change):.6f} eV\n")
    
    # Get optimized positions
    optimized_positions = atoms.get_positions()  # in Angstrom
    
    print("Initial geometry (Angstrom):")
    print(positions_angstrom)
    print()
    
    print("Optimized geometry (Angstrom):")
    print(optimized_positions)
    print()
    
    # Calculate displacements
    displacements = np.linalg.norm(optimized_positions - positions_angstrom, axis=1)
    max_displacement = displacements.max()
    avg_displacement = displacements.mean()
    rmsd = np.sqrt(np.mean(displacements**2))
    
    print(f"RMSD between initial and optimized: {rmsd:.6f} Angstrom")
    print(f"Maximum atomic displacement: {max_displacement:.6f} Angstrom")
    print(f"Average atomic displacement: {avg_displacement:.6f} Angstrom")
    print()
    
    ############################################
    # Save Results
    ############################################
    
    # Save optimized geometry to XYZ file
    output_file = Path(__file__).resolve().parent / "optimized_geometry_ase.xyz"
    ase_write(str(output_file), atoms, format="xyz")
    print(f"Optimized geometry saved to: {output_file}")
    
    # Save trajectory
    print(f"Optimization trajectory saved to: {trajectory_file}")
    print(f"Optimization log saved to: {logfile}")
    print()
    
    # Check if energy is NaN
    if np.isnan(final_energy):
        print("WARNING: Final energy is NaN!")
        return 1
    
    print("=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    
    return 0 if optimization_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
