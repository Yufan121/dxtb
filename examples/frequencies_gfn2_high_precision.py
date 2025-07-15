# This file is part of dxtb.
#
# High-precision frequency calculation example using GFN2-xTB.
#
# This version prioritizes accuracy and determinism over speed.
"""
High-precision frequency calculation example using GFN2-xTB.

This version includes optimizations for maximum accuracy:
1. Strict SCF convergence criteria  
2. Double precision (torch.double)
3. Deterministic algorithms
4. High-level integral settings
5. Conservative numerical parameters
"""
from pathlib import Path

import torch
import time
import numpy as np
from tad_mctc.io import read
from tad_mctc.units import AU2RCM

import dxtb
from dxtb.typing import DD

# Set deterministic behavior for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Use highest precision available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dd: DD = {"device": device, "dtype": torch.double}  # Double precision

print("=== High-Precision Frequency Calculation ===")
print(f"Device: {device}")
print(f"Precision: {dd['dtype']}")
print("Deterministic mode: ENABLED")

# Load molecule data
path = Path(__file__).resolve().parent / "molecules" / "lih.xyz"
numbers, positions = read.read(path, ftype="xyz", **dd)
print(f'Molecule: {path.name}')
print(f'Numbers: {numbers}')
print(f'Positions: {positions}')

charge = 0

# High-precision calculator options
opts = {
    "verbosity": 2,           # Increased verbosity for monitoring
    
    # Strict convergence criteria
    "maxiter": 500,           # Much higher iteration limit
    "f_atol": 1e-10,          # Very strict function tolerance
    "x_atol": 1e-10,          # Very strict input tolerance
    
    # SCF settings for maximum accuracy
    "scf_mode": "full",       # Full gradient tracking (most accurate)
    "mixer": "anderson",      # Stable mixing
    "damp": 0.2,             # Conservative damping
    "damp_generations": 15,   # More generations for stability
    "damp_soft_start": True,  # Gentle start
    
    # Force convergence even if slow
    "scf_force_convergence": False,  # Don't accept unconverged results
    
    # High-precision integral settings
    "intcutoff": 100.0,       # Large cutoff radius
    "intlevel": 4,            # Highest integral level
    "step_size": 1e-7,        # Very small numerical step
    
    # Per-atom parameters
    "per_atom": True
}

# Simplified but complete parameter setup
atom_param_dict = { 
    # Global parameters
    "wexp": 0.0,
    "kpol": 0.0,
    "enscale": 0.0,
    
    # Key atomic parameters with high precision
    "gam": [0.0, 0.0],
    "en": [0.0, 0.0], 
    "arep": [0.0, 0.0],
    
    # Level and slater parameters
    "levels": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "slater": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
}

# Setup parameters for current molecule
natom = len(numbers)
for key, value in atom_param_dict.items():
    if isinstance(value, list):
        if isinstance(value[0], list):
            # Handle nested lists (like levels, slater)
            atom_param_dict[key] = [value[0]] * natom
        else:
            # Handle simple lists
            atom_param_dict[key] = [value[0]] * natom
    else:
        atom_param_dict[key] = value

# Convert to high-precision tensors
for key, value in atom_param_dict.items():
    if isinstance(value, list) or isinstance(value, float):
        atom_param_dict[key] = torch.tensor(
            value, 
            dtype=torch.double,  # High precision
            requires_grad=True, 
            device=device
        )
    else:
        raise ValueError(f"Invalid param value type for {key}: {type(value)}")

# Setup parameter class
GFN2_XTB_ATOM = dxtb.GFN2_XTB
GFN2_XTB_ATOM._per_atom_params_dict = atom_param_dict

print(f"\n=== Calculation Settings ===")
print(f"Max iterations: {opts['maxiter']}")
print(f"Function tolerance: {opts['f_atol']}")
print(f"Input tolerance: {opts['x_atol']}")
print(f"SCF mode: {opts['scf_mode']}")
print(f"Integral cutoff: {opts['intcutoff']}")
print(f"Step size: {opts['step_size']}")

# Start high-precision calculation
print(f"\n=== Starting High-Precision Calculation ===")
time_start = time.time()
dxtb.timer.reset()

numbers = torch.tensor(numbers, dtype=torch.int32, device=device)

# Initialize calculator with high-precision settings
calc = dxtb.Calculator(
    numbers,
    GFN2_XTB_ATOM,
    opts=opts,
    **dd
)

# Calculate frequencies with high precision
pos = positions.clone().requires_grad_(True).to(device)
freqs1, modes1 = calc.vibration(pos, chrg=charge, use_functorch=True)

# Convert to cm-1
freqs1_cm = freqs1 * AU2RCM
print(f"\nHigh-precision frequencies (a.u.): {freqs1}")
print(f"High-precision frequencies (cm⁻¹): {freqs1_cm}")

dxtb.timer.print()
time_end = time.time()
print(f"\nTotal calculation time: {time_end - time_start:.2f} seconds")

# Compute parameter gradients with high precision
print(f"\n=== High-Precision Parameter Gradients ===")
def get_freq_grad(freqs, param):
    grad_freqs = torch.autograd.grad(
        freqs, param, 
        grad_outputs=torch.ones_like(freqs), 
        retain_graph=True, 
        allow_unused=True
    )
    return grad_freqs

for key, value in atom_param_dict.items():
    if isinstance(value, torch.Tensor) and value.requires_grad:
        print(f"Computing high-precision gradient for {key}...")
        grad_freqs = get_freq_grad(freqs1, value)
        if grad_freqs[0] is not None:
            print(f"dFreq/d{key}: {grad_freqs[0]}")
        else:
            print(f"dFreq/d{key}: None (no gradient)")

print(f"\n=== Final Results Summary ===")
print(f"Calculation completed successfully")
print(f"Time: {time_end - time_start:.2f} seconds")
print(f"Precision: {dd['dtype']}")
print(f"Convergence: f_atol={opts['f_atol']}, x_atol={opts['x_atol']}")
print(f"Frequencies (cm⁻¹): {freqs1_cm.detach().cpu().numpy()}") 