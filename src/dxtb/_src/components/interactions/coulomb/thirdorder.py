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
Coulomb: On-site third-order electrostatic energy (ES3)
=======================================================

This module implements the third-order electrostatic energy for GFN1-xTB.

Example
-------

.. code-block:: python

    import torch
    import dxtb.coulomb.thirdorder as es3
    from dxtb import GFN1_XTB, get_element_param
    from dxtb import IndexHelper

    # Define atomic numbers and their positions
    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [+0.00000000000000, -0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ])

    # Atomic charges
    qat = torch.tensor([
        -8.41282505804719e-2,
        2.10320626451180e-2,
        2.10320626451178e-2,
        2.10320626451179e-2,
        2.10320626451179e-2,
    ])

    # Initialize the ES3 calculation class with Hubbard derivatives parameter
    hubbard_derivs = get_element_param(GFN1_XTB.element, "gam3")
    es = es3.ES3(positions, hubbard_derivs)

    # Create an index helper from atomic numbers
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    # Generate the cache and carry out the energy calculation
    cache = es.get_cache(ihelp)
    e = es.get_atom_energy(qat, cache)

    # Print the summed energy
    torch.set_printoptions(precision=7)
    print(torch.sum(e, dim=-1))  # tensor(0.0155669)
"""

from __future__ import annotations

import torch
from tad_mctc.exceptions import DeviceError

from dxtb import IndexHelper
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import (
    DD,
    Any,
    Slicers,
    Tensor,
    TensorLike,
    get_default_dtype,
    override,
)

from ..base import Interaction, InteractionCache

__all__ = ["ES3", "LABEL_ES3", "new_es3"]


LABEL_ES3 = "ES3"
"""Label for the :class:`.ES3` interaction, coinciding with the class name."""


class ES3Cache(InteractionCache, TensorLike):
    """
    Restart data for the :class:`.ES3` interaction.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    hd: Tensor
    """Spread Hubbard derivatives of all atoms (not only unique)."""

    shell_resolved: bool
    """Whether the third-order electrostatics are shell-resolved."""
    
    predicted_energy_shell: Tensor | None
    """Predicted energy contributions spread to shell level."""

    __slots__ = ["__store", "hd", "shell_resolved", "predicted_energy_shell"]

    def __init__(
        self,
        hd: Tensor,
        shell_resolved: bool = False,
        predicted_energy_shell: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else hd.device,
            dtype=dtype if dtype is None else hd.dtype,
        )
        self.hd = hd
        self.shell_resolved = shell_resolved
        self.predicted_energy_shell = predicted_energy_shell
        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        hd: Tensor
        """Spread Hubbard derivatives of all atoms (not only unique)."""
        
        predicted_energy_shell: Tensor | None
        """Predicted energy contributions spread to shell level."""

        def __init__(self, hd: Tensor, predicted_energy_shell: Tensor | None = None) -> None:
            self.hd = hd
            self.predicted_energy_shell = predicted_energy_shell

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(self.hd, self.predicted_energy_shell)

        slicer = slicers["shell"] if self.shell_resolved else slicers["atom"]
        self.hd = self.hd[[~conv, *slicer]]
        if self.predicted_energy_shell is not None:
            self.predicted_energy_shell = self.predicted_energy_shell[[~conv, *slicer]]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.hd = self.__store.hd
        self.predicted_energy_shell = self.__store.predicted_energy_shell


class ES3(Interaction):
    """
    On-site third-order electrostatic energy (:class:`.ES3`).
    """

    hubbard_derivs: Tensor
    """Hubbard derivatives of all atoms."""

    shell_scale: Tensor | None
    """
    Scaling factors for shell-resolved third-order electrostatics.

    In GFN2-xTB, this is a tensor of shape ``(3,)`` containing the scaling
    factors for the s, p, and d shells.

    :default: ``None``
    """

    __slots__ = ["hubbard_derivs", "shell_scale"]   # slots means that the variables are stored in the class as a fixed size array

    def __init__(
        self,
        hubbard_derivs: Tensor,
        shell_scale: Tensor | None = None,
        # Yufan added
        hubbard_derivs_peratom: Tensor | None = None,
        shell_scale_peratom: Tensor | None = None,
        qsh_peratom: Tensor | None = None,
        predicted_energy_peratom: Tensor | None = None,
        # Yufan added end
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype, qsh_peratom=qsh_peratom)
        self.hubbard_derivs = hubbard_derivs
        self.shell_scale = shell_scale
        self.hubbard_derivs_peratom = hubbard_derivs_peratom
        self.shell_scale_peratom = shell_scale_peratom
        self.predicted_energy_peratom = predicted_energy_peratom

    # pylint: disable=unused-argument
    @override
    def get_cache( 
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> ES3Cache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        ES3Cache
            Restart data for the interaction.

        Note
        ----
        If the :class:`.ES3` interaction is evaluated within the
        :class:`dxtb.components. InteractionList`, ``positions`` will be
        passed as an argument, too. Hence, it is necessary to absorb
        the ``positions`` in the signature of the function (also see
        :meth:`dxtb.components.Interaction.get_cache`).
        """
        if numbers is None:
            raise ValueError("Atomic numbers are required for ES3 cache.")
        if ihelp is None:
            raise ValueError("IndexHelper is required for ES3 cache.")

        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, ES3Cache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        # if the cache is built, store the cachevar for validation
        self._cachevars = cachvars


        if self.shell_scale is None:
            hd = ihelp.spread_uspecies_to_atom(self.hubbard_derivs)
        else:
            # Start with global shell scaling factors
            scale = ihelp.spread_ushell_to_shell(self.shell_scale[ihelp.unique_angular]) # atoms' shell
            
            # Add per-atom delta corrections if available (3rd_scale_s/p/d as deltas)
            if self.shell_scale_peratom is not None:
                # Extract s, p, d delta components from concatenated peratom tensor  
                # n_atoms = len(ihelp.shells_per_atom)
                # delta_s = self.shell_scale_peratom[:n_atoms]
                # delta_p = self.shell_scale_peratom[n_atoms:2*n_atoms] 
                # delta_d = self.shell_scale_peratom[2*n_atoms:3*n_atoms]
                
                # # Map per-atom deltas to shell-level using pure tensor operations (gradient-safe)
                # # Use ihelp mapping functions to ensure gradient preservation
                # delta_s_spread = ihelp.spread_atom_to_shell(delta_s)
                # delta_p_spread = ihelp.spread_atom_to_shell(delta_p) 
                # delta_d_spread = ihelp.spread_atom_to_shell(delta_d)
                
                # # Select appropriate delta based on shell angular momentum
                # # Create boolean masks for each angular momentum type
                # is_s = (ihelp.unique_angular == 0)
                # is_p = (ihelp.unique_angular == 1) 
                # is_d = (ihelp.unique_angular == 2)
                
                # # Combine deltas using masks (fully differentiable)
                # delta_scale = (delta_s_spread * is_s.float() + 
                #               delta_p_spread * is_p.float() + 
                #               delta_d_spread * is_d.float())
                # Final scale = global + per-atom deltas
                
                
                from dxtb._src.xtb.base import _flatten_peratom_to_shell
                delta_scale = _flatten_peratom_to_shell(
                    self.shell_scale_peratom, ihelp.shells_per_atom
                )

                assert scale.shape == delta_scale.shape, f"{scale.shape} != {delta_scale.shape}"
                                
                scale = scale + delta_scale
            
            # scale = scale with softplus
            scale = torch.nn.functional.relu(scale)
            
            
            # ** Yufan added **
            # new way
            if self.hubbard_derivs_peratom is not None:
                hd_peratom = self.hubbard_derivs_peratom 
                hd_peratom_spread = ihelp.spread_atom_to_shell(hd_peratom)
                hd_element_spread = ihelp.spread_uspecies_to_shell(self.hubbard_derivs)
                
                assert hd_peratom_spread.shape == hd_element_spread.shape, f"{hd_peratom_spread.shape} != {hd_element_spread.shape}"
                
                # 验证映射正确性 (可选的调试检查)
                # assert hd_peratom.shape[0] == len(ihelp.shells_per_atom), f"Per-atom params length {hd_peratom.shape[0]} != n_atoms {len(ihelp.shells_per_atom)}"
                
                hd = (hd_element_spread + hd_peratom_spread) * scale
            else:
                hd = ihelp.spread_uspecies_to_shell(self.hubbard_derivs) * scale
            # ** Yufan added end **
            
        # Handle predicted energy spreading from atoms to shells
        predicted_energy_shell = None
        if self.predicted_energy_peratom is not None:
            from dxtb._src.xtb.base import _flatten_peratom_to_shell
            predicted_energy_shell = _flatten_peratom_to_shell(
                self.predicted_energy_peratom, ihelp.shells_per_atom
            )
            
        self.cache = ES3Cache(
            hd, shell_resolved=(self.shell_scale is not None), 
            predicted_energy_shell=predicted_energy_shell, **self.dd
        )
        
        def check_for_nan(tensor: Tensor, tensor_name: str) -> None:
            """Check if a tensor contains NaN values and raise an error if it does."""
            if torch.isnan(tensor).any():
                raise ValueError(f"{tensor_name} tensor contains NaN values.")

        check_for_nan(hd, "ES3.hd")

        return self.cache

    @override
    def get_monopole_atom_energy(
        self, cache: ES3Cache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Implements Eq.30 of the following paper:

        - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht,
          J. Seibert, S. Spicher and S. Grimme, *WIREs Computational Molecular
          Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493
          <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

        Parameters
        ----------
        cache : ES3Cache
            Restart data for the interaction.
        charges : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atom-wise third-order Coulomb interaction energies.
        """
        return (
            cache.hd * torch.pow(qat, 3.0) / 3.0
            if self.shell_scale is None
            else torch.zeros_like(qat)
        )

    @override
    def get_monopole_shell_energy(
        self, cache: ES3Cache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Parameters
        ----------
        cache : ES3Cache
            Restart data for the interaction.
        qat : Tensor
            Shell charges of all atoms.

        Returns
        -------
        Tensor
            Shell-wise third-order Coulomb interaction energy.
        """
        base_energy = (
            torch.zeros_like(qat)
            if self.shell_scale is None
            else cache.hd * torch.pow(qat, 3.0) / 3.0
        )
        
        # Add predicted energy contribution if available
        if cache.predicted_energy_shell is not None:
            base_energy = base_energy + cache.predicted_energy_shell
            
        return base_energy

    @override
    def get_monopole_atom_potential(
        self,
        cache: ES3Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate the third-order electrostatic potential.
        Zero if this interaction is shell-resolved.

        Parameters
        ----------
        qat : ES3Cache
            Restart data for the interaction.
        charges : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atom-wise third-order Coulomb interaction potential.
        """
        return (
            cache.hd * torch.pow(qat, 2.0)
            if self.shell_scale is None
            else torch.zeros_like(qat)
        )

    @override
    def get_monopole_shell_potential(
        self, cache: ES3Cache, qsh: Tensor, *_: Any, **__: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic potential.
        Zero if this interaction is atom-resolved.

        Parameters
        ----------
        qsh : Tensor
            Shell charges of all atoms.
        cache : ES3Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Shell-wise third-order Coulomb interaction potential.
        """
        base_potential = (
            torch.zeros_like(qsh)
            if self.shell_scale is None
            else cache.hd * torch.pow(qsh, 2.0)
        )
        # Add predicted energy contribution if available (keep shape)
        if cache.predicted_energy_shell is not None:
            base_potential = base_potential + cache.predicted_energy_shell
        return base_potential


def new_es3(
    unique: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ES3 | None:
    """
    Create new instance of :class:`.ES3`.

    Parameters
    ----------
    unique : Tensor
        Unique elements in the system (shape: ``(nunique,)``).
    par : Param | ParamModule
        Representation of an extended tight-binding model.

    Returns
    -------
    ES3 | None
        Instance of the :class:`.ES3` class or ``None`` if no :class:`.ES3` is
        used.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    # compatibility with previous version based on `Param`
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "thirdorder" not in par or par.is_none("thirdorder"):
        return None

    if device is not None:
        if device != unique.device:
            raise DeviceError(
                f"Passed device ({device}) and device of `unique` tensor "
                f"({unique.device}) do not match."
            )

    hubbard_derivs = par.get_elem_param(unique, "gam3") # only unique elements are considered
    hubbard_derivs_peratom = par.get_atom_param(unique, "gam3")
    # print(f"hubbard_derivs: {hubbard_derivs}")
    # print(f"hubbard_derivs_peratom: {hubbard_derivs_peratom}")

    shell_scale = (     # global parameter(s)
        None
        if par.is_false("thirdorder", "shell")
        else torch.cat(
            [
                torch.atleast_1d(par.get("thirdorder.shell.s")),
                torch.atleast_1d(par.get("thirdorder.shell.p")),
                torch.atleast_1d(par.get("thirdorder.shell.d")),
            ],
            dim=0,
        )
    )
    
    # Try to get per-atom shell scaling deltas (3rd_scale_s/p/d as corrections to global shell_scale)
    try:
        shell_scale_peratom = (
            par.get_atom_param(unique, "3rd_scale")
        )       

    except:
        shell_scale_peratom = None  # Fallback: use only global shell_scale
        
        
    try: 
        qsh_peratom = par.get_atom_param(unique, "qsh")
    except:
        qsh_peratom = None
        
    try:
        predicted_energy_peratom = par.get_atom_param(unique, "predicted_energy")
    except:
        predicted_energy_peratom = None
        
        
    # if shell_scale_peratom is not None:
    #     print(f"shell_scale_peratom: {shell_scale_peratom.shape}")
        
    return ES3(hubbard_derivs, shell_scale=shell_scale, 
               # Yufan added
               hubbard_derivs_peratom=hubbard_derivs_peratom, shell_scale_peratom=shell_scale_peratom,
               qsh_peratom=qsh_peratom, predicted_energy_peratom=predicted_energy_peratom,
               # Yufan added end
               **dd)
