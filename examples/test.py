import torch
import dxtb

dd = {"dtype": torch.double, "device": torch.device("cpu")}

# LiH
numbers = torch.tensor([3, 1, 4], device=dd["device"])
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]],
                         **dd)

# Yufan modification
tensor = torch.tensor([1.0, 2.0, 3.0], **dd)

# instantiate a calculator
calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

# compute the energy
pos = positions.clone().requires_grad_(True)
energy = calc.get_energy(pos)

# obtain gradient (dE/dR) via autograd
(g,) = torch.autograd.grad(energy, pos)


# Alternatively, forces can directly be requested from the calculator.
# (Don't forget to manually reset the calculator when the inputs are identical.)
calc.reset() 
pos = positions.clone().requires_grad_(True)
forces = calc.get_forces(pos)

assert torch.equal(forces, -g)



print(f"derivative of energy w.r.t. positions: {g}")