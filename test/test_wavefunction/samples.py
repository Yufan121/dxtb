"""Molecules for testing properties."""

from __future__ import annotations
import torch

from xtbml.typing import Tensor, Molecule
from xtbml.utils import symbol2number


class Record(Molecule):
    """Format of reference records (calculated with xTB 6.5.1)."""

    density: Tensor
    """Density matrix for GFN1-xTB."""

    n_electrons: Tensor
    """Number of valence electrons of molecule."""

    charges: Tensor
    """Charge of molecule (total or resolved)."""

    emo: Tensor
    """Orbital energies (taken from first iteration)."""

    e_fermi: Tensor
    """Fermi energy (taken from first iteration)."""

    entropy: Tensor
    """Electronic entropy at 300 K."""

    focc: Tensor
    """Fractional occupation at 300 K."""

    mulliken_charges: Tensor
    """Atom-resolved Mulliken partial charges."""

    mulliken_charges_shell: Tensor
    """Shell-resolved Mulliken partial charges."""

    mulliken_pop: Tensor
    """Shell-resolved Mulliken populations."""

    overlap: Tensor
    """Overlap matrix for GFN1-xTB."""

    wiberg: Tensor
    """Reference values for Wiberg bond orders."""


samples: dict[str, Record] = {
    "H2": {
        "numbers": symbol2number(["H", "H"]),
        "positions": torch.tensor(
            [
                0.00000000000000,
                0.00000000000000,
                -0.70252931147690,
                0.00000000000000,
                0.00000000000000,
                0.70252931147690,
            ],
        ).reshape((-1, 3)),
        "n_electrons": torch.tensor(2.0),
        "charges": torch.tensor(0.0),
        "density": torch.tensor(
            [
                0.59854788165593265,
                3.1933160084198133e-003,
                0.59854788165593242,
                3.1933160084199668e-003,
                3.1933160084198133e-003,
                1.7036677335518522e-005,
                3.1933160084198125e-003,
                1.7036677335519342e-005,
                0.59854788165593242,
                3.1933160084198125e-003,
                0.59854788165593231,
                3.1933160084199664e-003,
                3.1933160084199668e-003,
                1.7036677335519342e-005,
                3.1933160084199664e-003,
                1.7036677335520162e-005,
            ]
        ).reshape((4, 4)),
        "overlap": torch.tensor(
            [
                1.0000000000000000,
                0.0000000000000000,
                0.66998297071517465,
                6.5205745680663674e-002,
                0.0000000000000000,
                1.0000000000000000,
                6.5205745680664340e-002,
                0.10264305272175234,
                0.66998297071517465,
                6.5205745680664340e-002,
                1.0000000000000000,
                0.0000000000000000,
                6.5205745680663674e-002,
                0.10264305272175234,
                0.0000000000000000,
                1.0000000000000000,
            ]
        ).reshape((4, 4)),
        "e_fermi": torch.tensor([-0.3110880516469479]),
        "entropy": torch.tensor([0.0000000000000000]),
        "emo": torch.tensor(
            [
                -0.5292992591857910,
                -0.0928768441081047,
                -0.0746814981102943,
                0.2626818716526031,
            ]
        ),
        "focc": torch.tensor(
            [
                1.0000000000000000e00,
                1.7731362462295247e-100,
                8.5328773404458140e-109,
                5.1478349564629082e-263,
            ]
        ),
        "mulliken_charges": torch.tensor([0.00000, 0.00000]),
        "mulliken_charges_shell": torch.tensor(
            [
                2.2700790464325049e-004,
                -2.2700790464299256e-004,
                2.2700790464325049e-004,
                -2.2700790464297445e-004,
            ]
        ),
        "mulliken_pop": torch.tensor(
            [
                0.99977299209535675,
                2.2700790464299256e-004,
                0.99977299209535675,
                2.2700790464297445e-004,
            ]
        ),
        "wiberg": torch.tensor(
            [
                [0.0000000000000000, 1.0000000000000007],
                [1.0000000000000007, 0.0000000000000000],
            ]
        ),
    },
    "LiH": {
        "numbers": symbol2number(["Li", "H"]),
        "positions": torch.tensor(
            [
                0.00000000000000,
                0.00000000000000,
                -1.50796743897235,
                0.00000000000000,
                0.00000000000000,
                1.50796743897235,
            ],
        ).reshape((-1, 3)),
        "n_electrons": torch.tensor(2.0),
        "charges": torch.tensor(0.0),
        "density": torch.tensor(
            [
                0.20683869182353645,
                0.0000000000000000,
                0.0000000000000000,
                0.18337511649124147,
                0.43653303718019015,
                7.8111260740060754e-003,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.18337511649124147,
                0.0000000000000000,
                0.0000000000000000,
                0.16257322579116207,
                0.38701316392721402,
                6.9250397066456734e-003,
                0.43653303718019015,
                0.0000000000000000,
                0.0000000000000000,
                0.38701316392721402,
                0.92130292872059771,
                1.6485380751645420e-002,
                7.8111260740060754e-003,
                0.0000000000000000,
                0.0000000000000000,
                6.9250397066456734e-003,
                1.6485380751645420e-002,
                2.9498199783660944e-004,
            ]
        ).reshape((6, 6)),
        "overlap": torch.tensor(
            [
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.40560703697368278,
                -0.20099517781353604,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.46387245581706105,
                -7.5166937761294322e-002,
                0.40560703697368278,
                0.0000000000000000,
                0.0000000000000000,
                0.46387245581706105,
                1.0000000000000000,
                0.0000000000000000,
                -0.20099517781353604,
                0.0000000000000000,
                0.0000000000000000,
                -7.5166937761294322e-002,
                0.0000000000000000,
                1.0000000000000000,
            ]
        ).reshape((6, 6)),
        "emo": torch.tensor(
            [
                -0.4501341879367828,
                -0.2407598942518234,
                -0.1693878620862961,
                -0.1693877726793289,
                -0.0793312489986420,
                0.2863629460334778,
            ]
        ),
        "e_fermi": torch.tensor([-0.3454470410943031]),
        "entropy": torch.tensor([0.0000000000000000]),
        "focc": torch.tensor(
            [
                1.0000000000000000e00,
                1.3937844053026810e-48,
                3.2945722200787142e-81,
                3.2942254567938899e-81,
                2.2389049326083313e-122,
                1.5125388537358539e-289,
            ]
        ),
        "mulliken_charges": torch.tensor([0.27609, -0.27609]),
        "mulliken_charges_shell": torch.tensor(
            [
                0.61767599778670401,
                -0.34157592431816841,
                -0.27789560584675255,
                1.7955323782162611e-003,
            ]
        ),
        "mulliken_pop": torch.tensor(
            [
                0.38232400221329599,
                0.34157592431816841,
                1.2778956058467525,
                -1.7955323782162611e-003,
            ]
        ),
        "wiberg": torch.tensor(
            [
                [0.0000000000000000, 0.92377265725501168],
                [0.92377265725501168, 0.0000000000000000],
            ]
        ),
    },
    "SiH4": {
        "numbers": symbol2number(["Si", "H", "H", "H", "H"]),
        "positions": torch.tensor(
            [
                0.00000000000000,
                -0.00000000000000,
                0.00000000000000,
                1.61768389755830,
                1.61768389755830,
                -1.61768389755830,
                -1.61768389755830,
                -1.61768389755830,
                -1.61768389755830,
                1.61768389755830,
                -1.61768389755830,
                1.61768389755830,
                -1.61768389755830,
                1.61768389755830,
                1.61768389755830,
            ],
        ).reshape((-1, 3)),
        "n_electrons": torch.tensor(8.0),
        "charges": torch.tensor(0.0),
        "density": torch.tensor(
            [
                0.95266977814088072,
                -6.3188240095752813e-016,
                5.8478535363473266e-016,
                4.8228286168368446e-016,
                -1.5321944884146509e-031,
                2.0860667943490419e-025,
                1.6927390197181194e-017,
                -3.6499432127887788e-016,
                1.4516662171400914e-016,
                0.23137623597272830,
                6.7969218173179208e-003,
                0.23137623597272880,
                6.7969218173179781e-003,
                0.23137623597272938,
                6.7969218173181481e-003,
                0.23137623597272874,
                6.7969218173176095e-003,
                -6.3188240095752813e-016,
                0.34736135543141627,
                7.6327832942979512e-017,
                6.3143934525555778e-016,
                -1.0002303429944862e-017,
                1.0589079977801452e-018,
                -1.4224732503009818e-016,
                3.4694469519536142e-018,
                -0.11969117413304249,
                0.27325141025959709,
                -3.2189847434777753e-003,
                -0.27325141025959809,
                3.2189847434778108e-003,
                0.27325141025959804,
                -3.2189847434774951e-003,
                -0.27325141025959732,
                3.2189847434777475e-003,
                5.8478535363473266e-016,
                7.6327832942979512e-017,
                0.34736135543141561,
                4.1633363423443370e-016,
                2.2403404607595847e-016,
                5.6833727007981764e-019,
                1.2490009027033011e-016,
                -0.11969117413304239,
                3.2265856653168612e-016,
                0.27325141025959732,
                -3.2189847434778221e-003,
                -0.27325141025959776,
                3.2189847434775220e-003,
                -0.27325141025959648,
                3.2189847434780797e-003,
                0.27325141025959765,
                -3.2189847434777102e-003,
                4.8228286168368446e-016,
                6.3143934525555778e-016,
                4.1633363423443370e-016,
                0.34736135543141611,
                -2.3373850371649520e-016,
                -7.9186192927219019e-019,
                -0.11969117413304264,
                0.0000000000000000,
                -1.5959455978986625e-016,
                -0.27325141025959737,
                3.2189847434776113e-003,
                -0.27325141025959732,
                3.2189847434776677e-003,
                0.27325141025959720,
                -3.2189847434778061e-003,
                0.27325141025959748,
                -3.2189847434777714e-003,
                -1.5321944884146509e-031,
                -1.0002303429944862e-017,
                2.2403404607595847e-016,
                -2.3373850371649520e-016,
                3.0206292770933252e-031,
                8.6890495109000881e-034,
                8.0539862919355916e-017,
                -7.7195973591547266e-017,
                3.4465187990708944e-018,
                3.5223794868856983e-016,
                -4.1494701960557157e-018,
                1.5502300903535370e-017,
                -1.8262182087148494e-019,
                -3.6797454963937003e-016,
                4.3348521427647150e-018,
                2.3430004726511437e-019,
                -2.7601258372843535e-021,
                2.0860667943490419e-025,
                1.0589079977801452e-018,
                5.6833727007981764e-019,
                -7.9186192927219019e-019,
                8.6890495109000881e-034,
                5.9630660779252999e-036,
                2.7285385257704187e-019,
                -1.9583339970255441e-019,
                -3.6487058670001874e-019,
                1.9029879416593321e-018,
                -2.2417774809023393e-020,
                -6.5715329068212422e-019,
                7.7414677735717830e-021,
                -2.3701032611168600e-019,
                2.7920558318493872e-021,
                -1.0088241222071404e-018,
                1.1884257156908338e-020,
                1.6927390197181194e-017,
                -1.4224732503009818e-016,
                1.2490009027033011e-016,
                -0.11969117413304264,
                8.0539862919355916e-017,
                2.7285385257704187e-019,
                4.1242288301051075e-002,
                -9.0205620750793969e-017,
                2.9490299091605721e-017,
                9.4154924306018753e-002,
                -1.1091736528511399e-003,
                9.4154924306018170e-002,
                -1.1091736528511533e-003,
                -9.4154924306018475e-002,
                1.1091736528512088e-003,
                -9.4154924306018267e-002,
                1.1091736528511930e-003,
                -3.6499432127887788e-016,
                3.4694469519536142e-018,
                -0.11969117413304239,
                0.0000000000000000,
                -7.7195973591547266e-017,
                -1.9583339970255441e-019,
                -9.0205620750793969e-017,
                4.1242288301050957e-002,
                -1.2143064331837650e-016,
                -9.4154924306018475e-002,
                1.1091736528512101e-003,
                9.4154924306018323e-002,
                -1.1091736528511063e-003,
                9.4154924306018170e-002,
                -1.1091736528513016e-003,
                -9.4154924306018392e-002,
                1.1091736528511694e-003,
                1.4516662171400914e-016,
                -0.11969117413304249,
                3.2265856653168612e-016,
                -1.5959455978986625e-016,
                3.4465187990708944e-018,
                -3.6487058670001874e-019,
                2.9490299091605721e-017,
                -1.2143064331837650e-016,
                4.1242288301050950e-002,
                -9.4154924306017962e-002,
                1.1091736528511897e-003,
                9.4154924306018184e-002,
                -1.1091736528512021e-003,
                -9.4154924306018725e-002,
                1.1091736528510985e-003,
                9.4154924306018531e-002,
                -1.1091736528511876e-003,
                0.23137623597272830,
                0.27325141025959709,
                0.27325141025959732,
                -0.27325141025959737,
                3.5223794868856983e-016,
                1.9029879416593321e-018,
                9.4154924306018753e-002,
                -9.4154924306018475e-002,
                -9.4154924306017962e-002,
                0.70105339039614190,
                -5.9458540562029724e-003,
                -0.15875823851836182,
                4.1829885108348105e-003,
                -0.15875823851836068,
                4.1829885108354376e-003,
                -0.15875823851836154,
                4.1829885108348435e-003,
                6.7969218173179208e-003,
                -3.2189847434777753e-003,
                -3.2189847434778221e-003,
                3.2189847434776113e-003,
                -4.1494701960557157e-018,
                -2.2417774809023393e-020,
                -1.1091736528511399e-003,
                1.1091736528512101e-003,
                1.1091736528511897e-003,
                -5.9458540562029724e-003,
                1.3798398225090867e-004,
                4.1829885108350673e-003,
                1.8663135337776167e-005,
                4.1829885108347975e-003,
                1.8663135337773551e-005,
                4.1829885108347125e-003,
                1.8663135337776211e-005,
                0.23137623597272880,
                -0.27325141025959809,
                -0.27325141025959776,
                -0.27325141025959732,
                1.5502300903535370e-017,
                -6.5715329068212422e-019,
                9.4154924306018170e-002,
                9.4154924306018323e-002,
                9.4154924306018184e-002,
                -0.15875823851836182,
                4.1829885108350673e-003,
                0.70105339039614156,
                -5.9458540562027755e-003,
                -0.15875823851836079,
                4.1829885108344948e-003,
                -0.15875823851836132,
                4.1829885108347802e-003,
                6.7969218173179781e-003,
                3.2189847434778108e-003,
                3.2189847434775220e-003,
                3.2189847434776677e-003,
                -1.8262182087148494e-019,
                7.7414677735717830e-021,
                -1.1091736528511533e-003,
                -1.1091736528511063e-003,
                -1.1091736528512021e-003,
                4.1829885108348105e-003,
                1.8663135337776167e-005,
                -5.9458540562027755e-003,
                1.3798398225090531e-004,
                4.1829885108350499e-003,
                1.8663135337782306e-005,
                4.1829885108345807e-003,
                1.8663135337779104e-005,
                0.23137623597272938,
                0.27325141025959804,
                -0.27325141025959648,
                0.27325141025959720,
                -3.6797454963937003e-016,
                -2.3701032611168600e-019,
                -9.4154924306018475e-002,
                9.4154924306018170e-002,
                -9.4154924306018725e-002,
                -0.15875823851836068,
                4.1829885108347975e-003,
                -0.15875823851836079,
                4.1829885108350499e-003,
                0.70105339039614090,
                -5.9458540562030279e-003,
                -0.15875823851836146,
                4.1829885108347568e-003,
                6.7969218173181481e-003,
                -3.2189847434774951e-003,
                3.2189847434780797e-003,
                -3.2189847434778061e-003,
                4.3348521427647150e-018,
                2.7920558318493872e-021,
                1.1091736528512088e-003,
                -1.1091736528513016e-003,
                1.1091736528510985e-003,
                4.1829885108354376e-003,
                1.8663135337773551e-005,
                4.1829885108344948e-003,
                1.8663135337782306e-005,
                -5.9458540562030279e-003,
                1.3798398225091488e-004,
                4.1829885108349285e-003,
                1.8663135337777064e-005,
                0.23137623597272874,
                -0.27325141025959732,
                0.27325141025959765,
                0.27325141025959748,
                2.3430004726511437e-019,
                -1.0088241222071404e-018,
                -9.4154924306018267e-002,
                -9.4154924306018392e-002,
                9.4154924306018531e-002,
                -0.15875823851836154,
                4.1829885108347125e-003,
                -0.15875823851836132,
                4.1829885108345807e-003,
                -0.15875823851836146,
                4.1829885108349285e-003,
                0.70105339039614123,
                -5.9458540562030635e-003,
                6.7969218173176095e-003,
                3.2189847434777479e-003,
                -3.2189847434777102e-003,
                -3.2189847434777714e-003,
                -2.7601258372843505e-021,
                1.1884257156908338e-020,
                1.1091736528511930e-003,
                1.1091736528511694e-003,
                -1.1091736528511876e-003,
                4.1829885108348435e-003,
                1.8663135337776214e-005,
                4.1829885108347802e-003,
                1.8663135337779104e-005,
                4.1829885108347568e-003,
                1.8663135337777064e-005,
                -5.9458540562030635e-003,
                1.3798398225090480e-004,
            ]
        ).reshape(17, 17),
        "overlap": torch.tensor(
            [
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.42641210131950702,
                -0.14556746567207857,
                0.42641210131950702,
                -0.14556746567207857,
                0.42641210131950702,
                -0.14556746567207857,
                0.42641210131950702,
                -0.14556746567207857,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.28196829739436963,
                6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                0.28196829739436963,
                6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.28196829739436963,
                6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                0.28196829739436963,
                6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                0.28196829739436963,
                6.4855251203039338e-002,
                0.28196829739436963,
                6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.26132497245657471,
                0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                -0.26132497245657471,
                -0.12150281739592229,
                -0.26132497245657471,
                -0.12150281739592229,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                -0.26132497245657471,
                -0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                -0.26132497245657471,
                -0.12150281739592229,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                -0.26132497245657471,
                -0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                -0.26132497245657471,
                -0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                0.42641210131950702,
                0.28196829739436963,
                0.28196829739436963,
                -0.28196829739436963,
                0.0000000000000000,
                0.0000000000000000,
                0.26132497245657471,
                -0.26132497245657471,
                -0.26132497245657471,
                1.0000000000000000,
                0.0000000000000000,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                -0.14556746567207857,
                6.4855251203039338e-002,
                6.4855251203039338e-002,
                -6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                0.12150281739592229,
                -0.12150281739592229,
                -0.12150281739592229,
                0.0000000000000000,
                1.0000000000000000,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                0.42641210131950702,
                -0.28196829739436963,
                -0.28196829739436963,
                -0.28196829739436963,
                0.0000000000000000,
                0.0000000000000000,
                0.26132497245657471,
                0.26132497245657471,
                0.26132497245657471,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                1.0000000000000000,
                0.0000000000000000,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                -0.14556746567207857,
                -6.4855251203039338e-002,
                -6.4855251203039338e-002,
                -6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                0.12150281739592229,
                0.12150281739592229,
                0.12150281739592229,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                0.0000000000000000,
                1.0000000000000000,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                0.42641210131950702,
                0.28196829739436963,
                -0.28196829739436963,
                0.28196829739436963,
                0.0000000000000000,
                0.0000000000000000,
                -0.26132497245657471,
                0.26132497245657471,
                -0.26132497245657471,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                1.0000000000000000,
                0.0000000000000000,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                -0.14556746567207857,
                6.4855251203039338e-002,
                -6.4855251203039338e-002,
                6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                -0.12150281739592229,
                0.12150281739592229,
                -0.12150281739592229,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                0.0000000000000000,
                1.0000000000000000,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                0.42641210131950702,
                -0.28196829739436963,
                0.28196829739436963,
                0.28196829739436963,
                0.0000000000000000,
                0.0000000000000000,
                -0.26132497245657471,
                -0.26132497245657471,
                0.26132497245657471,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                1.0000000000000000,
                0.0000000000000000,
                -0.14556746567207857,
                -6.4855251203039338e-002,
                6.4855251203039338e-002,
                6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                -0.12150281739592229,
                -0.12150281739592229,
                0.12150281739592229,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                0.0000000000000000,
                1.0000000000000000,
            ]
        ).reshape((17, 17)),
        "emo": torch.tensor(
            [
                -0.6158187985420227,
                -0.4695263206958771,
                -0.4695262610912323,
                -0.4695262312889099,
                -0.1725806295871735,
                -0.1725805550813675,
                -0.1725803762674332,
                -0.0903786197304726,
                -0.0903783515095711,
                -0.0675993412733078,
                -0.0675990283489227,
                -0.0675989910960197,
                -0.0500356517732143,
                0.3943647742271423,
                1.2486594915390015,
                1.2486598491668701,
                1.2486605644226074,
            ]
        ),
        "e_fermi": torch.tensor([-0.3210534304380417]),
        "entropy": torch.tensor([0.0000000000000000]),
        "focc": torch.tensor(
            [
                1.0000000000000000e00,
                1.0000000000000000e00,
                1.0000000000000000e00,
                1.0000000000000000e00,
                1.3439754976862796e-68,
                1.3439754976862796e-68,
                1.3436925979341226e-68,
                3.5584392189446305e-106,
                3.5576901862615917e-106,
                1.3744270436278056e-116,
                1.3739931020026147e-116,
                1.3739931020026147e-116,
                1.2860122892070556e-124,
                0.0000000000000000e00,
                0.0000000000000000e00,
                0.0000000000000000e00,
                0.0000000000000000e00,
            ]
        ),
        "mulliken_charges": torch.tensor(
            [0.27511, -0.06878, -0.06878, -0.06878, -0.06878]
        ),
        "mulliken_charges_shell": torch.tensor(
            [
                0.65663937010219842,
                3.5838834166484368e-002,
                -0.41737062303296546,
                -7.1859260070046282e-002,
                3.0823647611154144e-003,
                -7.1859260070044950e-002,
                3.0823647611152557e-003,
                -7.1859260070046727e-002,
                3.0823647611154179e-003,
                -7.1859260070045616e-002,
                3.0823647611153034e-003,
            ]
        ),
        "mulliken_pop": torch.tensor(
            [
                1.3433606298978016,
                1.9641611658335156,
                0.41737062303296546,
                1.0718592600700463,
                -3.0823647611154144e-003,
                1.0718592600700450,
                -3.0823647611152557e-003,
                1.0718592600700467,
                -3.0823647611154179e-003,
                1.0718592600700456,
                -3.0823647611153034e-003,
            ]
        ),
        "wiberg": torch.tensor(
            [
                [
                    0.0000000000000000,
                    0.93865054674698889,
                    0.93865054674699089,
                    0.93865054674699011,
                    0.93865054674699089,
                ],
                [
                    0.93865054674698889,
                    0.0000000000000000,
                    1.7033959988060572e-002,
                    1.7033959988060138e-002,
                    1.7033959988060534e-002,
                ],
                [
                    0.93865054674699089,
                    1.7033959988060572e-002,
                    0.0000000000000000,
                    1.7033959988060347e-002,
                    1.7033959988060482e-002,
                ],
                [
                    0.93865054674699011,
                    1.7033959988060138e-002,
                    1.7033959988060347e-002,
                    0.0000000000000000,
                    1.7033959988060454e-002,
                ],
                [
                    0.93865054674699089,
                    1.7033959988060534e-002,
                    1.7033959988060482e-002,
                    1.7033959988060454e-002,
                    0.0000000000000000,
                ],
            ]
        ),
    },
    "S2": {
        "numbers": symbol2number(["S", "S"]),
        "positions": torch.tensor(
            [
                [0.00000000000000, 0.00000000000000, -1.80281271474629],
                [0.00000000000000, 0.00000000000000, 1.80281271474629],
            ],
        ),
        "n_electrons": torch.tensor(12.0),
        "charges": torch.tensor(0.0),
        "density": torch.tensor([0.0]),
        "overlap": torch.tensor([0.0]),
        "e_fermi": torch.tensor([-0.38648638884647568]),
        "entropy": torch.tensor([0.0]),
        "emo": torch.tensor(
            [
                -0.91044152961192060,
                -0.85637829714691738,
                -0.52215404077561622,
                -0.49373935245513334,
                -0.49373935245513323,
                -0.38648638884647579,
                -0.38648638884647551,
                -0.23834743677590314,
                -6.9709071441550821e-002,
                -6.9709071441550780e-002,
                -6.7441589665420631e-002,
                -6.7441589665420187e-002,
                -5.4310235785884023e-002,
                -5.4310235785884009e-002,
                -3.6572329964911998e-002,
                1.6310109000505803e-002,
                1.6310109000506279e-002,
                0.45417011472823121,
            ]
        ),
        "focc": torch.tensor(
            [
                1.0000000000000000e00,
                1.0000000000000000e00,
                1.0000000000000000e00,
                1.0000000000000000e00,
                1.0000000000000000e00,
                5.0000000000000000e-01,
                5.0000000000000000e-01,
                1.8680225545649920e-68,
                1.4689097425274104e-145,
                1.4689097425274104e-145,
                1.3468101442471582e-146,
                1.3468101442471582e-146,
                1.3401964658258330e-152,
                1.3401964658258330e-152,
                1.0414870660814827e-160,
                6.9909285390079426e-185,
                6.9909285390079426e-185,
                0.0000000000000000e00,
            ]
        ),
        "mulliken_charges": torch.tensor([0.00000, 0.00000]),
        "mulliken_charges_shell": torch.tensor([0.0]),
        "mulliken_pop": torch.tensor([0.0]),
        "wiberg": torch.tensor([0.0]),
    },
}
