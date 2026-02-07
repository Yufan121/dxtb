#!/usr/bin/env python3
"""
分子动力学（MD）模拟框架
使用 ASE（Atomic Simulation Environment）包装神经网络模型进行MD计算

使用方法：
    python md.py

依赖：
    pip install ase torch numpy
"""

import torch
import torch.nn as nn
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms, units
from ase.md import VelocityVerlet
from ase.optimize import BFGS
from ase.io import Trajectory
import matplotlib.pyplot as plt


class DummyNeuralNetwork(nn.Module):
    """
    简单的 dummy 神经网络模型
    用于演示如何将神经网络包装为 ASE 计算器
    """
    
    def __init__(self, n_atoms=3, hidden_dim=64):
        super().__init__()
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        
        # 简单的全连接网络
        self.energy_net = nn.Sequential(
            nn.Linear(n_atoms * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 力的网络（每个原子3个分量）
        self.force_net = nn.Sequential(
            nn.Linear(n_atoms * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_atoms * 3)
        )
    
    def forward(self, positions, numbers=None):
        """
        前向传播
        Args:
            positions: [batch_size, n_atoms, 3] 原子坐标
            numbers: [batch_size, n_atoms] 原子序数（可选）
        """
        batch_size = positions.shape[0]
        n_atoms = positions.shape[1]
        
        # 确保positions需要梯度
        if not positions.requires_grad:
            positions = positions.requires_grad_(True)
        
        # 展平坐标
        pos_flat = positions.view(batch_size, -1)
        
        # 计算能量
        energy = self.energy_net(pos_flat)
        
        # 计算力（通过自动微分）
        if positions.requires_grad:
            energy_grad = torch.autograd.grad(
                energy.sum(), positions, create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if energy_grad is not None:
                forces = -energy_grad  # 力是能量的负梯度
            else:
                # 如果梯度为None，使用力的网络
                forces = self.force_net(pos_flat).view(batch_size, n_atoms, 3)
        else:
            # 如果不需要梯度，使用力的网络
            forces = self.force_net(pos_flat).view(batch_size, n_atoms, 3)
        
        # set energies and forces to 0 tensors of the same shape as energy and forces
        energy = torch.zeros_like(energy)
        forces = torch.zeros_like(forces)
        
        return {
            'energy': energy.squeeze(-1),
            'forces': forces
        }


class NeuralNetworkCalculator(Calculator):
    """将神经网络包装为ASE计算器"""
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, model, device='cpu', **kwargs):
        """
        Args:
            model: 训练好的神经网络模型
            device: 计算设备 ('cuda' 或 'cpu')
        """
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """核心计算方法"""
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # 从ASE原子对象获取坐标和原子类型
        if atoms is None:
            return
        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        
        # 转换为模型需要的格式
        positions_tensor = torch.FloatTensor(positions).unsqueeze(0).to(self.device).requires_grad_(True)
        numbers_tensor = torch.LongTensor(numbers).unsqueeze(0).to(self.device)
        
        # 调用模型进行预测（不使用no_grad，因为模型内部需要计算梯度）
        result = self.model(positions_tensor, numbers_tensor)
        
        # 提取结果
        energy = result['energy'].item()
        forces = result['forces'].detach().cpu().numpy()[0]
        
        # 存储结果
        self.results = {
            'energy': energy,
            'forces': forces
        }


def create_water_molecule():
    """创建水分子结构"""
    # 水分子坐标（H-O-H键角104.5度，O-H键长0.957 Å）
    positions = [
        [0.0, 0.0, 0.0],  # O
        [0.957, 0.0, 0.0],  # H1
        [0.957 * np.cos(104.5 * np.pi / 180), 
         0.957 * np.sin(104.5 * np.pi / 180), 0.0]  # H2
    ]
    
    atoms = Atoms('OHH', positions=positions)
    atoms.set_cell([10, 10, 10])  # 模拟盒子
    atoms.set_pbc(True)  # 周期性边界条件
    
    return atoms


def run_energy_minimization(atoms, calculator, fmax=0.05):
    """运行能量最小化"""
    print("开始能量最小化...")
    atoms.calc = calculator
    
    opt = BFGS(atoms)
    opt.run(fmax=fmax)
    
    print(f"最小化完成，最终能量: {atoms.get_potential_energy():.6f} eV")
    return atoms


def run_molecular_dynamics(atoms, calculator, steps=1000, timestep=1.0, temperature=300):
    """运行分子动力学模拟"""
    print(f"开始MD模拟: {steps}步, 步长{timestep}fs, 温度{temperature}K")
    
    atoms.calc = calculator
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    
    # 设置初始温度
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    # 创建MD积分器
    dyn = VelocityVerlet(atoms, timestep * units.fs)
    
    # 创建轨迹文件
    trajectory = Trajectory('md_trajectory.traj', 'w')
    
    # 存储数据
    energies = []
    temperatures = []
    
    # 运行MD
    for step in range(steps):
        dyn.run(1)
        trajectory.write(atoms)
        
        energy = atoms.get_potential_energy()
        temp = atoms.get_temperature()
        
        energies.append(energy)
        temperatures.append(temp)
        
        if step % 100 == 0:
            print(f'Step {step:4d}: E={energy:8.4f} eV, T={temp:6.1f} K')
    
    trajectory.close()
    print("MD模拟完成！")
    
    return energies, temperatures


def plot_results(energies, temperatures):
    """绘制结果"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 能量图
    ax1.plot(energies)
    ax1.set_xlabel('MD Steps')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title('Potential Energy vs Time')
    ax1.grid(True)
    
    # 温度图
    ax2.plot(temperatures)
    ax2.set_xlabel('MD Steps')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Temperature vs Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('md_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=== 神经网络分子动力学模拟框架 ===\n")
    
    # 1. 创建 dummy 模型
    print("1. 创建 dummy 神经网络模型...")
    model = DummyNeuralNetwork(n_atoms=3, hidden_dim=64)
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 创建计算器
    print("2. 创建 ASE 计算器...")
    calculator = NeuralNetworkCalculator(model, device='cpu')
    
    # 3. 创建分子体系
    print("3. 创建水分子体系...")
    atoms = create_water_molecule()
    print(f"   原子数: {len(atoms)}")
    print(f"   化学式: {atoms.get_chemical_formula()}")
    
    # 4. 能量最小化
    print("\n4. 运行能量最小化...")
    atoms_opt = run_energy_minimization(atoms.copy(), calculator)
    
    # 5. 分子动力学模拟
    print("\n5. 运行分子动力学模拟...")
    energies, temperatures = run_molecular_dynamics(
        atoms_opt, calculator, 
        steps=500, 
        timestep=0.5, 
        temperature=300
    )
    
    # 6. 绘制结果
    print("\n6. 绘制结果...")
    plot_results(energies, temperatures)
    
    print("\n=== 模拟完成 ===")
    print("输出文件:")
    print("  - md_trajectory.traj: MD轨迹文件")
    print("  - md_results.png: 结果图表")


if __name__ == "__main__":
    main()
