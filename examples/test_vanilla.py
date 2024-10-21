import torch
import dxtb

dd = {"dtype": torch.double, "device": torch.device("cpu")}

# LiH
numbers = torch.tensor([3, 1, 4], device=dd["device"])
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]],
                         **dd)

#### Yufan modification
tensor_zeff = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, **dd)

tensor_dict = {"zeff": tensor_zeff}

# instantiate a calculator
calc = dxtb.calculators.GFN1Calculator(numbers, 
                                       **dd)

# compute the energy
pos = positions.clone().requires_grad_(True)
energy = calc.get_energy(pos)

# obtain gradient (dE/dR) via autograd
(g,) = torch.autograd.grad(energy, pos, retain_graph=True)

#### Yufan modification, get dE / dZeff
# (g_zeff,) = torch.autograd.grad(energy, tensor_zeff)

# Alternatively, forces can directly be requested from the calculator.
# (Don't forget to manually reset the calculator when the inputs are identical.)
calc.reset() 
pos = positions.clone().requires_grad_(True)
forces = calc.get_forces(pos)

assert torch.equal(forces, -g)



print(f"derivative of energy w.r.t. positions: {g}")
### YUFAN MODIFICATION
# print(f"derivative of energy w.r.t. zeff: {g_zeff}")
# delete the calculator
del calc


# ### Yufan modification
# # test fixed param output
# print(f"\n")
# print(f"Fixed param testing start")
# calc1 = dxtb.calculators.GFN1Calculator(numbers, **dd)
# pos = positions.clone().requires_grad_(True)
# energy1 = calc1.get_energy(pos)
# forces1 = calc1.get_forces(pos)

# # output RMSE
# print(f"RMSE of energy: {torch.sqrt(torch.mean((energy - energy1) ** 2))}")
# print(f"RMSE of forces: {torch.sqrt(torch.mean((forces - forces1) ** 2))}")