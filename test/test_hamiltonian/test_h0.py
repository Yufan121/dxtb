"""
Run tests for Hamiltonian.
"""

from __future__ import annotations
import pytest
import torch

from xtbml.adjlist import AdjacencyList
from xtbml.basis.type import get_cutoff
from xtbml.cutoff import get_lattice_points
from xtbml.exlibs.tbmalt import Geometry, batch
from xtbml.integral import mmd
from xtbml.ncoord.ncoord import get_coordination_number, exp_count
from xtbml.param.gfn1 import GFN1_XTB as par
from xtbml.typing import Tensor
from xtbml.xtb.calculator import Calculator
from xtbml.xtb.h0 import Hamiltonian

from .samples import mb16_43, Record


ovlp_sih4 = torch.tensor(
    [
        [
            1.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.4264,
            -0.1456,
            0.4264,
            -0.1456,
            0.4264,
            -0.1456,
            0.4264,
            -0.1456,
        ],
        [
            0.0000,
            1.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.2820,
            0.0649,
            -0.2820,
            -0.0649,
            0.2820,
            0.0649,
            -0.2820,
            -0.0649,
        ],
        [
            0.0000,
            0.0000,
            1.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.2820,
            0.0649,
            -0.2820,
            -0.0649,
            -0.2820,
            -0.0649,
            0.2820,
            0.0649,
        ],
        [
            0.0000,
            0.0000,
            0.0000,
            1.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            -0.2820,
            -0.0649,
            -0.2820,
            -0.0649,
            0.2820,
            0.0649,
            0.2820,
            0.0649,
        ],
        [
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            1.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],
        [
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            1.0000,
            0.0000,
            0.0000,
            0.0000,
            -0.2613,
            -0.1215,
            0.2613,
            0.1215,
            0.2613,
            0.1215,
            -0.2613,
            -0.1215,
        ],
        [
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            1.0000,
            0.0000,
            0.0000,
            -0.2613,
            -0.1215,
            0.2613,
            0.1215,
            -0.2613,
            -0.1215,
            0.2613,
            0.1215,
        ],
        [
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            1.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],
        [
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            1.0000,
            0.2613,
            0.1215,
            0.2613,
            0.1215,
            -0.2613,
            -0.1215,
            -0.2613,
            -0.1215,
        ],
        [
            0.4264,
            0.2820,
            0.2820,
            -0.2820,
            0.0000,
            -0.2613,
            -0.2613,
            0.0000,
            0.2613,
            1.0000,
            0.0000,
            0.0664,
            -0.0958,
            0.0664,
            -0.0958,
            0.0664,
            -0.0958,
        ],
        [
            -0.1456,
            0.0649,
            0.0649,
            -0.0649,
            0.0000,
            -0.1215,
            -0.1215,
            0.0000,
            0.1215,
            0.0000,
            1.0000,
            -0.0958,
            0.0271,
            -0.0958,
            0.0271,
            -0.0958,
            0.0271,
        ],
        [
            0.4264,
            -0.2820,
            -0.2820,
            -0.2820,
            0.0000,
            0.2613,
            0.2613,
            0.0000,
            0.2613,
            0.0664,
            -0.0958,
            1.0000,
            0.0000,
            0.0664,
            -0.0958,
            0.0664,
            -0.0958,
        ],
        [
            -0.1456,
            -0.0649,
            -0.0649,
            -0.0649,
            0.0000,
            0.1215,
            0.1215,
            0.0000,
            0.1215,
            -0.0958,
            0.0271,
            0.0000,
            1.0000,
            -0.0958,
            0.0271,
            -0.0958,
            0.0271,
        ],
        [
            0.4264,
            0.2820,
            -0.2820,
            0.2820,
            0.0000,
            0.2613,
            -0.2613,
            0.0000,
            -0.2613,
            0.0664,
            -0.0958,
            0.0664,
            -0.0958,
            1.0000,
            0.0000,
            0.0664,
            -0.0958,
        ],
        [
            -0.1456,
            0.0649,
            -0.0649,
            0.0649,
            0.0000,
            0.1215,
            -0.1215,
            0.0000,
            -0.1215,
            -0.0958,
            0.0271,
            -0.0958,
            0.0271,
            0.0000,
            1.0000,
            -0.0958,
            0.0271,
        ],
        [
            0.4264,
            -0.2820,
            0.2820,
            0.2820,
            0.0000,
            -0.2613,
            0.2613,
            0.0000,
            -0.2613,
            0.0664,
            -0.0958,
            0.0664,
            -0.0958,
            0.0664,
            -0.0958,
            1.0000,
            0.0000,
        ],
        [
            -0.1456,
            -0.0649,
            0.0649,
            0.0649,
            0.0000,
            -0.1215,
            0.1215,
            0.0000,
            -0.1215,
            -0.0958,
            0.0271,
            -0.0958,
            0.0271,
            -0.0958,
            0.0271,
            0.0000,
            1.0000,
        ],
    ]
)

ovlp_lih = torch.tensor(
    [
        [1.0000, 0.0000, 0.0000, 0.0000, 0.4056, -0.2010],
        [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000, 0.4639, -0.0752],
        [0.4056, 0.0000, 0.0000, 0.4639, 1.0000, 0.0000],
        [-0.2010, 0.0000, 0.0000, -0.0752, 0.0000, 1.0000],
    ]
)


class Setup:
    """Setup class to define constants for test class."""

    atol: Tensor = torch.tensor(1e-04)
    """Absolute tolerance for equality comparison in `torch.allclose`."""

    rtol: Tensor = torch.tensor(1e-04)
    """Relative tolerance for equality comparison in `torch.allclose`."""

    cutoff: Tensor = torch.tensor(30.0)
    """Cutoff for calculation of coordination number."""


class TestHamiltonian(Setup):
    """Testing the building of the Hamiltonian matrix."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

    def base_test(self, sample: Record, dtype: torch.dtype) -> None:
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        mol = Geometry(numbers, positions)

        # TODO: extend geometry object with mol.lattice and mol.periodic
        mol_periodic = [False]
        mol_lattice = None

        # setup calculator
        calc = Calculator(mol, par)

        # prepate cutoffs and lattice
        cutoff = get_cutoff(calc.basis)
        trans = get_lattice_points(mol_periodic, mol_lattice, cutoff)
        adjlist = AdjacencyList(mol, trans, cutoff)

        # build hamiltonian
        h, ovlp = calc.hamiltonian.build(calc.basis, adjlist, None)
        print("ovlp", ovlp)
        self.check_hamiltonian(h, ref)

    def base_test_cn(self, sample: Record, dtype: torch.dtype) -> None:
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        mol = Geometry(numbers, positions)

        # TODO: extend geometry object with mol.lattice and mol.periodic
        mol_periodic = [False]
        mol_lattice = None

        # setup calculator
        calc = Calculator(mol, par)

        # prepate cutoffs and lattice
        trans = get_lattice_points(mol_periodic, mol_lattice, self.cutoff)
        cn = get_coordination_number(numbers, positions, exp_count)

        cutoff = get_cutoff(calc.basis)
        trans = get_lattice_points(mol_periodic, mol_lattice, cutoff)
        adjlist = AdjacencyList(mol, trans, cutoff)

        # build hamiltonian
        h, _ = calc.hamiltonian.build(calc.basis, adjlist, cn)
        self.check_hamiltonian(h, ref)

    def check_hamiltonian(self, hamiltonian: Tensor, ref: Tensor) -> None:
        size = hamiltonian.size(dim=0)
        for i in range(size):
            for j in range(size):
                # print(i, j, hamiltonian[i, j], ref[i, j])
                diff = hamiltonian[i, j] - ref[i, j]
                tol = self.atol + self.rtol * torch.abs(ref[i, j])
                assert (
                    diff < tol
                ), f"Hamiltonian does not match reference.\nh[{i}, {j}] = {hamiltonian[i, j]} and ref[{i}, {j}] = {ref[i, j]} -> diff = {diff}"

    ##############
    #### GFN1 ####
    ##############

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def stest_hamiltonian_h2_gfn1(self, dtype: torch.dtype) -> None:
        """
        Compare against reference calculated with tblite-int: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1
        """

        self.base_test(mb16_43["H2"], dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def stest_hamiltonian_h2_gfn1_cn(self, dtype: torch.dtype) -> None:
        """
        Compare against reference calculated with tblite-int: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1 --cn 0.91396028097949444,0.91396028097949444
        """

        self.base_test_cn(mb16_43["H2_cn"], dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def stest_hamiltonian_lih_gfn1(self, dtype: torch.dtype) -> None:
        """
        Compare against reference calculated with tblite-int: fpm run -- Li H 0,0,3.0159348779447 --bohr --hamiltonian --method gfn1
        """

        sample = mb16_43["LiH"]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        # h0 = Hamiltonian(numbers, positions, par)
        # h = h0.build(ovlp_sih4)

        self.base_test(mb16_43["LiH"], dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def stest_hamiltonian_hli_gfn1(self, dtype: torch.dtype) -> None:
        """
        Compare against reference calculated with tblite-int: fpm run -- H Li 0,0,3.0159348779447 --bohr --hamiltonian --method gfn1
        """

        self.base_test(mb16_43["HLi"], dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def stest_hamiltonian_s2_gfn1(self, dtype: torch.dtype) -> None:
        """
        Compare against reference calculated with tblite-int: fpm run -- S S 0,0,3.60562542949258 --bohr --hamiltonian --method gfn1
        """

        self.base_test(mb16_43["S2"], dtype)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_hamiltonian_sih4_gfn1(self, dtype: torch.dtype) -> None:
        """
        Compare against reference calculated with tblite
        """
        # tblite: with "use dftd3_ncoord, only: get_coordination_number"

        sample = mb16_43["SiH4"]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        h0 = Hamiltonian(numbers, positions, par)
        h = h0.build(ovlp_sih4)

        print(torch.allclose(h, ref))
        # print("ref", ref / ovlp_sih4)
        self.check_hamiltonian(h, ref)

        # self.base_test(sample, ref_hamiltonian)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def stest_hamiltonian_sih4_gfn1_cn(self, dtype: torch.dtype) -> None:
        """
        Compare against reference calculated with tblite
        """
        # tblite: with "use dftd3_ncoord, only: get_coordination_number"

        sample = mb16_43["SiH4_cn"]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        h0 = Hamiltonian(numbers, positions, par)
        cn = get_coordination_number(numbers, positions, exp_count)
        h = h0.build(ovlp_sih4, cn)

        print(torch.allclose(h, ref))
        self.check_hamiltonian(h, ref)

        # self.base_test_cn(sample, dtype)

        # vec = torch.tensor([[-1.6177, -1.6177, 1.6177], [-1.6177, -1.6177, 1.6177]])
        # ang_i = torch.tensor([0, 0])
        # ang_j = torch.tensor([0, 0])
        # alpha_i = torch.tensor(
        #     [[7.6120, 1.3929, 0.3870, 0.1284], [7.6120, 1.3929, 0.3870, 0.1284]]
        # )
        # alpha_j = torch.tensor(
        #     [
        #         [7.5815, 2.1312, 0.8324, 0.2001, 0.1111, 0.0631],
        #         [7.5815, 2.1312, 0.8324, 0.2001, 0.1111, 0.0631],
        #     ]
        # )
        # coeff_i = torch.tensor(
        #     [[0.1854, 0.2377, 0.1863, 0.0446], [0.1854, 0.2377, 0.1863, 0.0446]]
        # )
        # coeff_j = torch.tensor(
        #     [
        #         [-0.0221, -0.0709, -0.0986, 0.1180, 0.0688, 0.0065],
        #         [-0.0221, -0.0709, -0.0986, 0.1180, 0.0688, 0.0065],
        #     ]
        # )

        # s = mmd.overlap((ang_i, ang_j), (alpha_i, alpha_j), (coeff_i, coeff_j), vec)
        # print(s)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_batch(self, dtype: torch.dtype) -> None:
        """
        Compare against reference calculated with tblite
        """
        # tblite: with "use dftd3_ncoord, only: get_coordination_number"

        sample1, sample2 = mb16_43["SiH4"], mb16_43["LiH"]

        numbers = batch.pack(
            (
                sample1["numbers"],
                sample2["numbers"],
            )
        )
        positions = batch.pack(
            (
                sample1["positions"].type(dtype),
                sample2["positions"].type(dtype),
            )
        )
        ref = batch.pack(
            (
                sample1["h0"].type(dtype),
                sample2["h0"].type(dtype),
            ),
        )
        ovlp = batch.pack((ovlp_sih4, ovlp_lih))

        h0 = Hamiltonian(numbers, positions, par)
        h = h0.build(ovlp)

        print(torch.allclose(h, ref, atol=1e-4, rtol=1e-4))

        for _batch in range(ref.shape[0]):
            self.check_hamiltonian(h[_batch, ...], ref[_batch, ...])

        # self.base_test(sample, ref_hamiltonian)

    ##############
    #### GFN2 ####
    ##############

    # def test_hamiltonian_h2_gfn2(self) -> None:
    #     nao = 2
    #     ref_hamiltonian = torch.tensor([
    #         -3.91986875628795E-1, -4.69784163992013E-1,
    #         -4.69784163992013E-1, -3.91986875628795E-1
    #     ]).reshape(nao, nao)

    #     self.base_test(data["MB16_43_H2"], ref_hamiltonian)

    # def test_hamiltonian_lih_gfn2(self) -> None:
    #     nao = 5
    #     ref_hamiltonian = torch.tensor([
    #         -1.85652586923456E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.04060196214555E-1, 0.00000000000000E+0,
    #         -7.93540972812401E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -7.93540972812401E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -7.93540972812401E-2, -2.64332062163992E-1, -2.04060196214555E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.64332062163992E-1,
    #         -3.91761139212137E-1
    #     ]).reshape(nao, nao)

    #     self.base_test(data["MB16_43_LiH"], ref_hamiltonian)

    # def test_hamiltonian_s2_gfn2(self) -> None:
    #     nao = 18
    #     ref_hamiltonian = torch.tensor([
    #         -7.35145147501899E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -1.92782969898654E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.36427116435023E-1, -2.05951870741313E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -4.17765757158496E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -9.33756556185781E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 1.20176381592757E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -4.17765757158496E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -9.33756556185781E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         1.20176381592757E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -4.17765757158496E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.36427116435023E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.58607478516679E-1, -1.23733824351312E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.27200169863736E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.05951870741313E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         1.23733824351312E-1, -9.40352474745915E-3, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.27200169863736E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -1.20176381592757E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 2.45142819855746E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.27200169863736E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.20176381592757E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.45142819855746E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.27200169863736E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -1.00344447956584E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.27200169863736E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.00344447956584E-2,
    #         -1.92782969898654E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.36427116435023E-1, -2.05951870741313E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -7.35145147501899E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -9.33756556185781E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.20176381592757E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -4.17765757158496E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -9.33756556185781E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -1.20176381592757E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -4.17765757158496E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.36427116435023E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.58607478516679E-1, 1.23733824351312E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -4.17765757158496E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.05951870741313E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -1.23733824351312E-1, -9.40352474745915E-3, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.27200169863736E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 1.20176381592757E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 2.45142819855746E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.27200169863736E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 1.20176381592757E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.45142819855746E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.27200169863736E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -1.00344447956584E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.27200169863736E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.00344447956584E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.27200169863736E-2
    #     ]).reshape(nao, nao)

    #     self.base_test(data["MB16_43_S2"], ref_hamiltonian)

    # def test_hamiltonian_sih4_gfn2(self) -> None:
    #     nao = 13
    #     ref_hamiltonian = torch.tensor([
    #         -5.52420992289823E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -3.36004309624475E-1, -3.36004309624475E-1, -3.36004309624475E-1,
    #         -3.36004309624475E-1, 0.00000000000000E+0, -2.35769689453471E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -1.53874693770200E-1, 1.53874693770200E-1,
    #         -1.53874693770200E-1, 1.53874693770200E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.35769689453471E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.53874693770200E-1,
    #         1.53874693770200E-1, 1.53874693770200E-1, -1.53874693770200E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.35769689453471E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         1.53874693770200E-1, 1.53874693770200E-1, -1.53874693770200E-1,
    #         -1.53874693770200E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -4.13801957898401E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -4.13801957898401E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 1.23912378305726E-1,
    #         -1.23912378305726E-1, -1.23912378305726E-1, 1.23912378305726E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -4.13801957898401E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         1.23912378305726E-1, -1.23912378305726E-1, 1.23912378305726E-1,
    #         -1.23912378305726E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -4.13801957898401E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -4.13801957898401E-2, -1.23912378305726E-1,
    #         -1.23912378305726E-1, 1.23912378305726E-1, 1.23912378305726E-1,
    #         -3.36004309624475E-1, -1.53874693770200E-1, -1.53874693770200E-1,
    #         1.53874693770200E-1, 0.00000000000000E+0, 1.23912378305726E-1,
    #         1.23912378305726E-1, 0.00000000000000E+0, -1.23912378305726E-1,
    #         -3.91823578951118E-1, -4.31486716382575E-2, -4.31486716382575E-2,
    #         -4.31486716382575E-2, -3.36004309624475E-1, 1.53874693770200E-1,
    #         1.53874693770200E-1, 1.53874693770200E-1, 0.00000000000000E+0,
    #         -1.23912378305726E-1, -1.23912378305726E-1, 0.00000000000000E+0,
    #         -1.23912378305726E-1, -4.31486716382575E-2, -3.91823578951118E-1,
    #         -4.31486716382575E-2, -4.31486716382575E-2, -3.36004309624475E-1,
    #         -1.53874693770200E-1, 1.53874693770200E-1, -1.53874693770200E-1,
    #         0.00000000000000E+0, -1.23912378305726E-1, 1.23912378305726E-1,
    #         0.00000000000000E+0, 1.23912378305726E-1, -4.31486716382575E-2,
    #         -4.31486716382575E-2, -3.91823578951118E-1, -4.31486716382575E-2,
    #         -3.36004309624475E-1, 1.53874693770200E-1, -1.53874693770200E-1,
    #         -1.53874693770200E-1, 0.00000000000000E+0, 1.23912378305726E-1,
    #         -1.23912378305726E-1, 0.00000000000000E+0, 1.23912378305726E-1,
    #         -4.31486716382575E-2, -4.31486716382575E-2, -4.31486716382575E-2,
    #         -3.91823578951118E-1
    #     ]).reshape(nao, nao)

    #     self.base_test(data["MB16_43_SiH4"], ref_hamiltonian)
