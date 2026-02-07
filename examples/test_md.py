#!/usr/bin/env python3
"""
简化的MD测试代码
"""

import torch
import torch.nn as nn
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms, units
from ase.md import VelocityVerlet
from ase.optimize import BFGS


class SimpleDummyModel(nn.Module):
    """简化的dummy模型"""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 1)  # 3原子 * 3坐标 = 9
    
    def forward(self, positions, numbers=None):
        batch_size = positions.shape[0]
        pos_flat = positions.view(batch_size, -1)
        energy = self.linear(pos_flat)
        
        # 计算力
        positions.requires_grad_(True)
        energy_grad = torch.autograd.grad(
            energy.sum(), positions, create_graph=True, retain_graph=True
        )[0]
        forces = -energy_grad
        
        return {
            'energy': energy.squeeze(-1),
            'forces': forces
        }


class SimpleCalculator(Calculator):
    """简化的计算器"""
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, model, device='cpu'):
        Calculator.__init__(self)
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        if atoms is None:
            return
            
        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        
        with torch.no_grad():
            positions_tensor = torch.FloatTensor(positions).unsqueeze(0).to(self.device)
            numbers_tensor = torch.LongTensor(numbers).unsqueeze(0).to(self.device)
            
            result = self.model(positions_tensor, numbers_tensor)
            
            energy = result['energy'].item()
            forces = result['forces'].cpu().numpy()[0]
        
        self.results = {
            'energy': energy,
            'forces': forces
        }


def test_md():
    """测试MD功能"""
    print("=== 测试神经网络MD框架 ===")
    
    # 1. 创建模型和计算器
    model = SimpleDummyModel()
    calculator = SimpleCalculator(model)
    
    # 2. 创建水分子
    atoms = Atoms('OHH', 
                  positions=[[0, 0, 0], [0.957, 0, 0], 
                           [0.957 * np.cos(104.5 * np.pi / 180), 
                            0.957 * np.sin(104.5 * np.pi / 180), 0]])
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(True)
    
    # 3. 设置计算器
    atoms.calc = calculator
    
    # 4. 测试能量计算
    print(f"初始能量: {atoms.get_potential_energy():.6f} eV")
    print(f"初始力: {atoms.get_forces()}")
    
    # 5. 能量最小化
    print("\n开始能量最小化...")
    opt = BFGS(atoms)
    opt.run(fmax=0.1)
    print(f"最小化后能量: {atoms.get_potential_energy():.6f} eV")
    
    # 6. 短时间MD测试
    print("\n开始短时间MD测试...")
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    atoms.set_temperature(300)
    
    dyn = VelocityVerlet(atoms, 0.5 * units.fs)
    
    for step in range(10):
        dyn.run(1)
        energy = atoms.get_potential_energy()
        temp = atoms.get_temperature()
        print(f'Step {step:2d}: E={energy:8.4f} eV, T={temp:6.1f} K')
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_md()
