#!/usr/bin/env python3
"""
Test script to verify the compare_gfn2.py functions work correctly.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compare_gfn2 import read_reference_data, convert_units_force_tensor

def test_file_reading():
    """Test reading the reference data files."""
    coords_file = "/scratch/kx58/yx7184/githubs/NNxTB/nnxtb_autograd/data_processing/output.xyz"
    forces_file = "/scratch/kx58/yx7184/githubs/NNxTB/nnxtb_autograd/data_processing/output_force.xyz"
    
    print("Testing file reading...")
    
    try:
        numbers, positions, ref_forces, ref_energy, charge, multiplicity = read_reference_data(
            coords_file, forces_file
        )
        
        print(f"✓ Successfully read data:")
        print(f"  Numbers shape: {numbers.shape}")
        print(f"  Positions shape: {positions.shape}")
        print(f"  Forces shape: {ref_forces.shape}")
        print(f"  Energy: {ref_energy} Hartree")
        print(f"  Charge: {charge}")
        print(f"  Multiplicity: {multiplicity}")
        
        # Test unit conversion
        ref_forces_converted = convert_units_force_tensor(ref_forces)
        print(f"  Forces converted shape: {ref_forces_converted.shape}")
        print(f"  Sample force (original): {ref_forces[0, 0].item():.6f} eV/Å")
        print(f"  Sample force (converted): {ref_forces_converted[0, 0].item():.6f} Hartree/Bohr")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading files: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_file_reading()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")
        sys.exit(1)
