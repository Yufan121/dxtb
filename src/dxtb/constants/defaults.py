"""
Default settings for `dxtb` calculations.
"""

import torch

# General

METHOD = "gfn1"
"""General method for calculation from the xtb family."""

METHOD_CHOICES = ["gfn1", "gfn1-xtb", "gfn2", "gfn2-xtb"]
"""List of possible choices for `METHOD`."""

SPIN = None
"""Total spin of the system."""

CHRG = 0
"""Total charge of the system."""

THRESH = {
    torch.float16: torch.tensor(1e-2, dtype=torch.float16),
    torch.float32: torch.tensor(1e-5, dtype=torch.float32),
    torch.float64: torch.tensor(1e-10, dtype=torch.float64),
}
"""Convergence thresholds for different float data types."""

EXCLUDE = []
"""List of xTB components to exclude during the calculation."""

EXCLUDE_CHOICES = ["disp", "rep", "hal", "es2", "es3", "scf", "all"]
"""List of possible choices for `EXCLUDE`."""


# SCF settings

GUESS = "eeq"
"""Initial guess for orbital charges."""

GUESS_CHOICES = ["eeq", "sad"]
"""List of possible choices for `GUESS`."""

MAXITER = 20
"""Maximum number of SCF iterations."""

VERBOSITY = 1
"""Verbosity of printout."""

XITORCH_XATOL = 1.0e-5
"""The absolute tolerance of the norm of the input of the equilibrium function."""

XITORCH_FATOL = 1.0e-5
"""The absolute tolerance of the norm of the output of the equilibrium function."""

# Fermi smearing

ETEMP = 300.0
"""Electronic temperature for Fermi smearing."""

FERMI_MAXITER = 200
"""Maximum number of iterations for Fermi smearing."""

FERMI_FENERGY_PARTITION = "equal"
"""Partitioning scheme for electronic free energy."""

FERMI_FENERGY_PARTITION_CHOICES = ["equal", "atomic"]
"""List of possible choices for `FERMI_FENERGY_PARTITION`."""
