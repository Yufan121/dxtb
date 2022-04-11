# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from xtbml.basis.type import Basis
from xtbml.exlibs.tbmalt import Geometry
from xtbml.param import Param
from xtbml.xtb.h0 import Hamiltonian


class Calculator:
    """
    Parametrized calculator defining the extended tight-binding model.

    The calculator holds the atomic orbital basis set for defining the Hamiltonian
    and the overlap matrix.
    """

    basis: Basis
    """Atomic orbital basis set definition."""

    hamiltonian: Hamiltonian
    """Core Hamiltonian definition."""

    def __init__(self, mol: Geometry, par: Param, acc: float = 1.0) -> None:

        self.basis = Basis(mol, par, acc)
        self.hamiltonian = Hamiltonian(mol, par)
