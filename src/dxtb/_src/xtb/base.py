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
xTB Hamiltonians: Base
======================

Base class for xTB Hamiltonians.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import symmetrize
from tad_mctc.data.radii import ATOMIC as ATOMIC_RADII
from tad_mctc.exceptions import DeviceError, DtypeError
from tad_mctc.units import EV2AU

from dxtb import IndexHelper
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import CNFunction, PathLike, Tensor, TensorLike

from .abc import HamiltonianABC

__all__ = ["BaseHamiltonian"]

PAD = -1


class BaseHamiltonian(HamiltonianABC, TensorLike):
    """
    Base class for GFN Hamiltonians.

    For the Hamiltonians, no integral driver is needed. Therefore, the
    signatures are different from the integrals over atomic orbitals. The most
    important difference is the `build` method, which does not require the
    driver anymore and only takes the positions (and the overlap integral).
    """

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""
    unique: Tensor
    """Unique species of the system."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    hscale: Tensor
    """Off-site scaling factor for the Hamiltonian."""
    kcn: Tensor
    """Coordination number dependent shift of the self energy."""
    kpair: Tensor
    """Element-pair-specific parameters for scaling the Hamiltonian."""
    refocc: Tensor
    """Reference occupation numbers."""
    selfenergy: Tensor
    """Self-energy of each species."""
    shpoly: Tensor
    """Polynomial parameters for the distant dependent scaling."""
    valence: Tensor
    """
    Whether the shell belongs to the valence shell.
    Only requried for GFN1-xTB (second s-function for H).
    """

    en: Tensor
    """Pauling electronegativity of each species."""
    enscale: Tensor
    """Electronegativity scaling factor."""
    rad: Tensor
    """Van-der-Waals radius of each species."""

    cn: CNFunction | None
    """Coordination number function."""

    __slots__ = [   # slots 是用于存储类属性的内存空间, 可以提高性能
        "numbers",
        "unique",
        "ihelp",
        "hscale",
        "kcn",
        "kpair",
        "refocc",
        "selfenergy",
        "shpoly",
        "valence",
        "en",
        "enscale",
        "rad",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_,
    ) -> None:
        super().__init__(device, dtype)

        # check device of input tensors
        if any(tensor.device != self.device for tensor in (numbers, ihelp)):
            raise ValueError("All input tensors must be on the same device")

        if not isinstance(par, ParamModule):
            par = ParamModule(par, **self.dd)

        if par.is_none("hamiltonian"):
            raise RuntimeError("Parametrization does not specify Hamiltonian.")

        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.ihelp = ihelp

        self.label = self.__class__.__name__
        self._matrix = None

        # Initialize Hamiltonian parameters

        # atom-resolved parameters
        self.rad = ATOMIC_RADII.to(**self.dd)[self.unique]                  # ATOMIC_RADII only used here 
        self.en = par.get_elem_param(self.unique, "en", pad_val=PAD)
        self.enscale = par.get("hamiltonian.xtb.enscale")

        # shell-resolved element parameters
        self.kcn = par.get_elem_param(self.unique, "kcn", pad_val=PAD)
        self.selfenergy = par.get_elem_param(self.unique, "levels", pad_val=PAD)
        self.shpoly = par.get_elem_param(self.unique, "shpoly", pad_val=PAD)
        self.refocc = par.get_elem_param(self.unique, "refocc", pad_val=PAD)
        self.valence = self._get_elem_valence(par)
        self.slater = par.get_elem_param(self.unique, "slater", pad_val=PAD)
        self.slater = self.ihelp.spread_ushell_to_shell(self.slater)
        
        ###### load per atom deltas
        self.rad_peratom = par.get_atom_param(self.unique, "arad")
        self.en_peratom = par.get_atom_param(self.unique, "en")
        self.kcn_peratom = par.get_atom_param(self.unique, "kcn")
        self.selfenergy_peratom = par.get_atom_param(self.unique, "levels")
        self.shpoly_peratom = par.get_atom_param(self.unique, "shpoly")
        # self.refocc_peratom = par.get_atom_param(self.unique, "refocc")

        
        ###### get pair parameters
        try:
            self.theta_ss_perpair = par.get_atom_param(self.unique, "theta_ss") # shape (natom * natom))
            self.theta_pp_perpair = par.get_atom_param(self.unique, "theta_pp")
            self.theta_dd_perpair = par.get_atom_param(self.unique, "theta_dd")
            self.theta_sp_perpair = par.get_atom_param(self.unique, "theta_sp")
            self.theta_sd_perpair = par.get_atom_param(self.unique, "theta_sd")
            self.theta_pd_perpair = par.get_atom_param(self.unique, "theta_pd")
            
            self.zeta_ss_perpair = par.get_atom_param(self.unique, "zeta_ss")
            self.zeta_pp_perpair = par.get_atom_param(self.unique, "zeta_pp")
            self.zeta_dd_perpair = par.get_atom_param(self.unique, "zeta_dd")
            self.zeta_sp_perpair = par.get_atom_param(self.unique, "zeta_sp")
            self.zeta_sd_perpair = par.get_atom_param(self.unique, "zeta_sd")
            self.zeta_pd_perpair = par.get_atom_param(self.unique, "zeta_pd")
            
            # make shell matrix
            self.theta_shell, self.zeta_shell = self._make_shell_matrix()
            
            self.slater_peratom = par.get_atom_param(self.unique, "slater")
            z = self.slater_peratom        
            z_list = []
            for i, n_shell in enumerate(self.ihelp.shells_per_atom):
                z_list.append(z[i, :n_shell])
            z = torch.cat(z_list)
            self.slater_peratom = z

            # flatten the use or not
            self.ml_mult = True
            
        except Exception:
            self.ml_mult = False



        # shell-pair-resolved pair parameters
        self.hscale = self._get_hscale(par) # 壳层间缩放因子矩阵, per ushell pair, # not used
        self.kpair = par.get_pair_param(self.unique.tolist())       # 怎么处理，目前全1，暂时不处理

        self.hscale_peratom = self._get_hscale_peratomshell(par) # 直接预测每个原子的hscale
        # TODO, kpair， pending for now, currently all 1
        
        # unit conversion
        self.selfenergy = self.selfenergy * EV2AU
        self.kcn = self.kcn * EV2AU
        
        # unit conversion for per atom parameters
        self.selfenergy_peratom = self.selfenergy_peratom * EV2AU
        self.kcn_peratom = self.kcn_peratom * EV2AU

        tensors = [
            ("hscale", self.hscale),
            ("kcn", self.kcn),
            ("kpair", self.kpair),
            ("refocc", self.refocc),
            ("selfenergy", self.selfenergy),
            ("shpoly", self.shpoly),
            ("en", self.en),
            ("rad", self.rad),
        ]

        for name, tensor in tensors:
            if tensor.dtype != self.dtype:
                raise DtypeError(
                    f"Tensor '{name}' has dtype '{tensor.dtype}'; "
                    f"expected '{self.dtype}'."
                )

        # For device checking, include an extra tensor 'valence'
        tensors_device = tensors + [("valence", self.valence)]
        for name, tensor in tensors_device:
            if tensor.device != self.device:
                raise DeviceError(
                    f"Tensor '{name}' is on device '{tensor.device}'; "
                    f"expected '{self.device}'."
                )

    def _make_shell_matrix(self) -> tuple[Tensor, Tensor]:
        """
        Make the shell matrix using self.ihelp.shells_to_ushell
        input: theta_ss_perpair, theta_pp_perpair, theta_dd_perpair, theta_sp_perpair, theta_sd_perpair, theta_pd_perpair (shape: (natom, natom))
        0: s, 1: p, 2: d (max is d)
        output: theta_shell, zeta_shell (shape: (shell, shell))
        """
        n_shells = len(self.ihelp.shells_to_ushell)
        theta_shell = torch.zeros((n_shells, n_shells), **self.dd)
        zeta_shell = torch.zeros((n_shells, n_shells), **self.dd)
        
        # Map angular momentum to parameter names
        angular_to_param = {
            (0, 0): ('ss', self.theta_ss_perpair, self.zeta_ss_perpair),  # s-s
            (1, 1): ('pp', self.theta_pp_perpair, self.zeta_pp_perpair),  # p-p
            (2, 2): ('dd', self.theta_dd_perpair, self.zeta_dd_perpair),  # d-d
            (0, 1): ('sp', self.theta_sp_perpair, self.zeta_sp_perpair),  # s-p
            (1, 0): ('sp', self.theta_sp_perpair, self.zeta_sp_perpair),  # p-s (same as s-p)
            (0, 2): ('sd', self.theta_sd_perpair, self.zeta_sd_perpair),  # s-d
            (2, 0): ('sd', self.theta_sd_perpair, self.zeta_sd_perpair),  # d-s (same as s-d)
            (1, 2): ('pd', self.theta_pd_perpair, self.zeta_pd_perpair),  # p-d
            (2, 1): ('pd', self.theta_pd_perpair, self.zeta_pd_perpair),  # d-p (same as p-d)
        }
        
        for i in range(n_shells):
            ush_i = self.ihelp.shells_to_ushell[i]
            ang_i = self.ihelp.unique_angular[ush_i]
            
            for j in range(n_shells):
                ush_j = self.ihelp.shells_to_ushell[j]
                ang_j = self.ihelp.unique_angular[ush_j]
                
                # Get the appropriate parameters based on angular momentum pair
                ang_pair = (int(ang_i), int(ang_j))
                if ang_pair in angular_to_param:
                    param_name, theta_param, zeta_param = angular_to_param[ang_pair]
                    
                    # Get the atom indices for shells i and j
                    atom_i = self.ihelp.shells_to_atom[i]
                    atom_j = self.ihelp.shells_to_atom[j]
                    
                    theta_shell[i, j] = theta_param[atom_i, atom_j]
                    zeta_shell[i, j] = zeta_param[atom_i, atom_j]
                else:
                    raise ValueError(f"Unsupported angular momentum pair: {ang_pair}")
            
        return theta_shell, zeta_shell




    @property
    def matrix(self) -> Tensor | None:
        """Hamiltonian matrix."""
        return self._matrix

    @matrix.setter
    def matrix(self, mat: Tensor) -> None:
        self._matrix = mat

    def clear(self) -> None:
        """Clear the integral matrix."""
        self._matrix = None

    @property
    def requires_grad(self) -> bool:
        """Whether the Hamiltonian matrix will be differentiated."""
        if self._matrix is None:
            return False

        return self._matrix.requires_grad

    def _get_elem_valence(self, _: ParamModule) -> Tensor:
        """
        Obtain a mask for valence and non-valence shells. This is only required
        for GFN1-xTB's second hydrogen s-function. For GFN2-xTB, this is a
        dummy method, i.e., the mask is always ``True``.

        Returns
        -------
        Tensor
            Mask indicating valence shells for each unique species.
        """
        return torch.ones(
            len(self.ihelp.unique_angular), device=self.device, dtype=torch.bool
        )

    def get_occupation(self) -> Tensor:
        """
        Obtain the reference occupation numbers for each orbital.
        """

        refocc = self.ihelp.spread_ushell_to_orbital(self.refocc)
        orb_per_shell = self.ihelp.spread_shell_to_orbital(
            self.ihelp.orbitals_per_shell
        )

        return torch.where(
            orb_per_shell != 0,
            refocc / orb_per_shell,
            torch.tensor(0, **self.dd),
        )

    def to_pt(self, path: PathLike | None = None) -> None:
        """
        Save the integral matrix to a file.

        Parameters
        ----------
        path : PathLike | None
            Path to the file where the integral matrix should be saved. If
            ``None``, the matrix is saved to the default location.
        """
        if path is None:
            path = f"{self.label.casefold()}.pt"

        torch.save(self.matrix, path)

    def build(self, positions: Tensor, overlap: Tensor | None = None) -> Tensor:
        """
        Build the xTB Hamiltonian.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        overlap : Tensor | None, optional
            Overlap matrix. If ``None``, the true xTB Hamiltonian is *not*
            built. Defaults to ``None``.

        Returns
        -------
        Tensor
            Hamiltonian (always symmetric).
        """
        # masks
        mask_atom_diagonal = real_pairs(self.numbers, mask_diagonal=True)
        mask_shell = real_pairs(
            self.ihelp.spread_atom_to_shell(self.numbers), mask_diagonal=False
        )
        mask_shell_diagonal = self.ihelp.spread_atom_to_shell(
            mask_atom_diagonal, dim=(-2, -1)
        )

        zero = torch.tensor(0.0, **self.dd)

        # ----------------
        # Eq.29: H_(mu,mu)
        # ----------------
        if self.cn is None:
            cn = torch.zeros_like(self.numbers, **self.dd)
        else:
            cn = self.cn(self.numbers, positions)   # TODO

        kcn = self.ihelp.spread_ushell_to_shell(self.kcn)
        kcn_peratom_list = []   # 获取每个原子的kcn
        for i, n_shell in enumerate(self.ihelp.shells_per_atom):
            kcn_peratom_list.append(self.kcn_peratom[i, :n_shell])
        kcn_peratom_flat = torch.cat(kcn_peratom_list)
        assert kcn_peratom_flat.shape == kcn.shape, f"kcn_peratom_flat.shape: {kcn_peratom_flat.shape}, kcn.shape: {kcn.shape}"
        kcn = kcn + kcn_peratom_flat
        
        # formula differs from paper to be consistent with GFN2 -> "kcn" adapted
        # selfenergy = self.ihelp.spread_ushell_to_shell(
        #     self.selfenergy
        # ) - kcn * self.ihelp.spread_atom_to_shell(cn)
        selfenergy_peratom_list = [] # 获取每个原子的selfenergy
        for i, n_shell in enumerate(self.ihelp.shells_per_atom):
            selfenergy_peratom_list.append(self.selfenergy_peratom[i, :n_shell])
        selfenergy_peratom_flat = torch.cat(selfenergy_peratom_list)
        selfenergy = self.ihelp.spread_ushell_to_shell(self.selfenergy)
        assert selfenergy_peratom_flat.shape == selfenergy.shape, f"selfenergy_peratom_flat.shape: {selfenergy_peratom_flat.shape}, selfenergy.shape: {selfenergy.shape}"
        selfenergy = selfenergy + selfenergy_peratom_flat
        selfenergy = selfenergy - kcn * self.ihelp.spread_atom_to_shell(cn)    ###### eq17

        # ----------------------
        # Eq.24: PI(R_AB, l, l')
        # ----------------------
        distances = storch.cdist(positions, positions, p=2)
        rad = self.ihelp.spread_uspecies_to_atom(self.rad)  # TODO
        # add rad_peratom
        assert self.rad_peratom.shape == rad.shape, f"self.rad_peratom.shape: {self.rad_peratom.shape}, rad.shape: {rad.shape}"
        rad = rad + self.rad_peratom
        
        rad = torch.nn.functional.relu(rad)

        rr = storch.divide(distances, rad.unsqueeze(-1) + rad.unsqueeze(-2))
        rr_shell = self.ihelp.spread_atom_to_shell(
            torch.where(mask_atom_diagonal, storch.sqrt(rr), zero),
            (-2, -1),
        )

        shpoly = self.ihelp.spread_ushell_to_shell(self.shpoly)
        shpoly_peratom_list = [] # 获取每个原子的shpoly
        for i, n_shell in enumerate(self.ihelp.shells_per_atom):
            shpoly_peratom_list.append(self.shpoly_peratom[i, :n_shell])
        shpoly_peratom_flat = torch.cat(shpoly_peratom_list)
        assert shpoly_peratom_flat.shape == shpoly.shape, f"shpoly_peratom_flat.shape: {shpoly_peratom_flat.shape}, shpoly.shape: {shpoly.shape}"
        shpoly = shpoly + shpoly_peratom_flat
        var_pi = (1.0 + shpoly.unsqueeze(-1) * rr_shell) * (  ###### eq 19
            1.0 + shpoly.unsqueeze(-2) * rr_shell
        )

        # --------------------
        # Eq.28: X(EN_A, EN_B)
        # --------------------
        en = self.ihelp.spread_uspecies_to_shell(self.en)
        en_peratom = self.ihelp.spread_atom_to_shell(self.en_peratom)
        assert en_peratom.shape == en.shape, f"en_peratom.shape: {en_peratom.shape}, en.shape: {en.shape}"
        en = en + en_peratom
        
        en = torch.nn.functional.relu(en)
        
        var_x = torch.where( # eq 16: (1 + k_en * (en_A - en_B)^2))
            mask_shell_diagonal,
            1.0
            + self.enscale
            * torch.pow(en.unsqueeze(-1) - en.unsqueeze(-2), 2.0),
            zero,
        )

        # --------------------
        # Eq.23: K_{AB}^{l,l'}
        # --------------------
        kpair = self.ihelp.spread_uspecies_to_shell(self.kpair, dim=(-2, -1))
        hscale = self.ihelp.spread_ushell_to_shell(self.hscale, dim=(-2, -1))
        valence = self.ihelp.spread_ushell_to_shell(self.valence)

        # assert self.hscale_peratom.shape == hscale.shape, f"self.hscale_peratom.shape: {self.hscale_peratom.shape}, hscale.shape: {hscale.shape}"
        hscale = self.hscale_peratom # no longer use hscale

        var_k = torch.where( # 
            valence.unsqueeze(-1) * valence.unsqueeze(-2),
            hscale * kpair * var_x,
            hscale,
        ) 

        # ------------
        # Eq.23: H_EHT ###### eq16 and eq17
        # ------------
        var_h = torch.where( # eq 16: 1/2 (H_kk + H_lmbd_lmbd)
            mask_shell,
            0.5 * (selfenergy.unsqueeze(-1) + selfenergy.unsqueeze(-2)),
            zero,
        )
        
        hcore_shell =   torch.where(
                            mask_shell_diagonal,
                            var_pi * var_k * var_h,  # scale only off-diagonals
                            var_h,
                        )
        
        
        
        
        if self.ml_mult:
            ###### additional multiplicative term for hcore ##### including the diagonal term START
            
            #### 1.0 get r_AB, duplicate for each shell pair
            # Use shells_per_atom to duplicate distances for each shell pair
            
            
            shells_per_atom = self.ihelp.shells_per_atom
            total_shells = shells_per_atom.sum()
            r_AB_shell = torch.zeros((total_shells, total_shells), **self.dd)
            # Create shell index mapping for efficient tensor operations
            shell_indices = torch.cumsum(torch.cat([torch.tensor([0], device=shells_per_atom.device), shells_per_atom[:-1]]), dim=0)
            for i, n_shells_i in enumerate(shells_per_atom):
                start_i = shell_indices[i]
                end_i = start_i + n_shells_i
                for j, n_shells_j in enumerate(shells_per_atom):
                    start_j = shell_indices[j]
                    end_j = start_j + n_shells_j
                    # Fill the block corresponding to atom pair (i,j)
                    r_AB_shell[start_i:end_i, start_j:end_j] = distances[i, j]  # correctness verified      
            
            #### 1.1 get theta, truncated in init: self.theta_shell, self.zeta_shell
            # Done in init

            #### 1.2 get r_AB * exp(- (ZETA + zeta_l + zeta_l') * r_AB)   
            # Get the total slater per shell
            slater = self.slater + self.slater_peratom
                
            # Create shell-wise slater matrix
            slater_i = slater.unsqueeze(-1)  # (n_shell, 1)
            slater_j = slater.unsqueeze(-2)  # (1, n_shell)
            
            # Calculate ZETA + zeta_l + zeta_l' term, ZETA is shell-pair matrix
            exponent_term = self.zeta_shell + slater_i + slater_j
            
            # Calculate r_AB * exp(- (ZETA + zeta_l + zeta_l') * r_AB)
            H_multi = r_AB_shell * torch.exp(-exponent_term * r_AB_shell)

            #### 1.3 get theta * r_AB * exp(- (ZETA + zeta_l + zeta_l') * r_AB) + 1
            H_multi = self.theta_shell * H_multi + 1
            
            
            #### 1.4 normalization factor
            # TODO

            #### 1.5 multiply
            hcore_shell = hcore_shell * H_multi

            ###### additional multiplicative term for hcore ##### including the diagonal term END
            
            # print hcore_shell stats and H_multi stats
            print(f"hcore_shell.min(): {hcore_shell.min()}, hcore_shell.max(): {hcore_shell.max()}")
            print(f"H_multi.min(): {H_multi.min()}, H_multi.max(): {H_multi.max()}")

        else:
            hcore_shell = hcore_shell


        print(f"hcore_shell.shape: {hcore_shell.shape}")
        # print(f"hcore_shell: {hcore_shell}")
        
        
        hcore = self.ihelp.spread_shell_to_orbital(   ##### TODO: add custom scaling after this hcore.
            hcore_shell,
            dim=(-2, -1),
        )
        
        # print(f"hcore.shape: {hcore.shape}") # (shell, shell)
        # print(f"hcore: {hcore}")


        if overlap is not None:
            hcore = hcore * overlap

        # force symmetry to avoid problems through numerical errors
        h0 = symmetrize(hcore)
        self.matrix = h0
        
        
        def check_for_nan(tensor: Tensor, tensor_name: str) -> None:
            """Check if a tensor contains NaN values and raise an error if it does."""
            if torch.isnan(tensor).any():
                raise ValueError(f"{tensor_name} tensor contains NaN values.")

        check_for_nan(h0, "h0")
        
        return h0
