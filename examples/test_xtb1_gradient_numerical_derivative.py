import torch
import dxtb


def numerical_derivative(func, inputs, param, h=1e-7):
    param_data = param.clone().detach()
    grad = torch.zeros_like(param_data)
    
    for i in range(len(param_data)):
        param_data[i] += h
        pos_plus = inputs.clone().detach().requires_grad_(True)
        energy_plus = func(pos_plus, param_data)
        
        param_data[i] -= 2 * h
        pos_minus = inputs.clone().detach().requires_grad_(True)
        energy_minus = func(pos_minus, param_data)
        
        grad[i] = (energy_plus - energy_minus) / (2 * h)
        
        param_data[i] += h  # Reset param_data[i]
    
    return grad

# Define a wrapper function to compute energy with given parameters
def compute_energy_with_params(pos, zeff, arep, en):
    tensor_dict = {"zeff": zeff, "arep": arep, "en": en}
    calc = dxtb.calculators.GFN1Calculator(numbers, tensor_dict=tensor_dict, **dd)
    energy = calc.get_energy(pos)
    calc.reset()
    return energy


dd = {"dtype": torch.double, "device": torch.device("cpu")}

# LiH
numbers = torch.tensor([3, 1, 4], device=dd["device"])
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]], **dd)

#### Yufan modification
tensor_zeff = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, **dd)
tensor_arep = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, **dd)
tensor_en = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, **dd)
tensor_dict = {"zeff": tensor_zeff, "arep": tensor_arep, "en": tensor_en}

# instantiate a calculator
calc = dxtb.calculators.GFN1Calculator(numbers, tensor_dict=tensor_dict, **dd)

# compute the energy
pos = positions.clone().requires_grad_(True)
energy = calc.get_energy(pos)

# obtain gradient (dE/dR) via autograd
(g,) = torch.autograd.grad(energy, pos, retain_graph=True)
#### Yufan modification, get dE / dZeff
(g_zeff,) = torch.autograd.grad(energy, tensor_zeff, retain_graph=True)
(g_arep,) = torch.autograd.grad(energy, tensor_arep, retain_graph=True)
(g_en,) = torch.autograd.grad(energy, tensor_en, retain_graph=True)

# Alternatively, forces can directly be requested from the calculator.
# (Don't forget to manually reset the calculator when the inputs are identical.)
calc.reset() 
pos = positions.clone().requires_grad_(True)
forces = calc.get_forces(pos)
assert torch.equal(forces, -g)

calc.reset() 

print(f"derivative of energy w.r.t. positions: {g}")
### YUFAN MODIFICATION
print(f"derivative of energy w.r.t. zeff: {g_zeff}, \nderivative of energy w.r.t. arep: {g_arep}, \nderivative of energy w.r.t. en: {g_en}")

# Numerical derivatives
grad_zeff = numerical_derivative(lambda p, z: compute_energy_with_params(p, z, tensor_arep, tensor_en), pos, tensor_zeff)
grad_arep = numerical_derivative(lambda p, a: compute_energy_with_params(p, tensor_zeff, a, tensor_en), pos, tensor_arep)
grad_en = numerical_derivative(lambda p, e: compute_energy_with_params(p, tensor_zeff, tensor_arep, e), pos, tensor_en)

print(f"Numerical derivative of energy w.r.t. zeff: {grad_zeff}")
print(f"Numerical derivative of energy w.r.t. arep: {grad_arep}")
print(f"Numerical derivative of energy w.r.t. en: {grad_en}")

# Comparison of autograd and numerical derivatives
print(f"Comparison of derivatives:")
print(f"Autograd derivative w.r.t. zeff: {g_zeff}")
print(f"Numerical derivative w.r.t. zeff: {grad_zeff}")
print(f"Difference: {g_zeff - grad_zeff}")

print(f"Autograd derivative w.r.t. arep: {g_arep}")
print(f"Numerical derivative w.r.t. arep: {grad_arep}")
print(f"Difference: {g_arep - grad_arep}")

print(f"Autograd derivative w.r.t. en: {g_en}")
print(f"Numerical derivative w.r.t. en: {grad_en}")
print(f"Difference: {g_en - grad_en}")

### Yufan modification
# test fixed param output
print(f"\n")
print(f"Fixed param testing start")
# calc1 = dxtb.calculators.GFN1Calculator(numbers, **dd)

# # explicitly reset tensor_dict
# from dxtb._src.calculators.types.base import reset_tensor_dict
# reset_tensor_dict()
import copy
calc1 = copy.deepcopy(dxtb.calculators.GFN1Calculator(numbers, tensor_dict=None, **dd))
# calc1.reset()
# import sys
# print(sys.modules['dxtb'])

positions1 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]], **dd)
pos1 = positions1.clone().requires_grad_(True)
energy1 = calc1.get_energy(pos)
forces1 = calc1.get_forces(pos)

# output RMSE
print(f"RMSE of energy: {torch.sqrt(torch.mean((energy - energy1) ** 2))}")
print(f"RMSE of forces: {torch.sqrt(torch.mean((forces - forces1) ** 2))}")
