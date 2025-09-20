#!/usr/bin/env python3
"""
使用torch.jit.fork并行计算DXTB力的简化版本
"""
import torch
import time
import random
from pathlib import Path
from tad_mctc.io import read
import dxtb


# 全局设置
device = torch.device("cpu")
dd = {"device": device, "dtype": torch.double}

def create_perturbed_params(numbers):
    """创建随机扰动的参数"""
    natom = len(numbers)
    
    atom_param_dict = {
        # 全局参数
        "wexp": 0.0, "kpol": 0.0, "enscale": 0.0, "ss": 0.0, "pp": 0.0, 
        "dd": 0.0, "sd": 0.0, "pd": 0.0, "s6": 0.0, "s8": 0.0,
        "a1": 0.0, "a2": 0.0, "s9": 0.0, "s10": 0.0, "kexp": 0.0, 
        "klight": 0.0, "gexp": 0.0, "s": 0.0, "p": 0.0, "d": 0.0,
        "dmp3": 0.0, "dmp5": 0.0, "shift": 0.0, "rmax": 0.0,
        
        # 原子参数
        "levels": [[0,0,0], [0,0,0]], 
        "slater": [[0.0,0.0,0.0], [0.0,0.0,0.0]],
        "shpoly": [[0,0,0], [0,0,0]], 
        "kcn": [[0,0,0], [0,0,0]],
        "gam": [0.0, 0.0], "lgam": [[0.0,0.0,0.0], [0.0,0.0,0.0]],
        "gam3": [0.0, 0.0], "zeff": [0, 0], "arep": [0, 0], "en": [0, 0],
        "dkernel": [0, 0], "qkernel": [0, 0], "mprad": [0, 0], "mpvcn": [0, 0],
        "3rd_scale": [[0, 0, 0], [0, 0, 0]], "rcov": [0, 0], "arad": [0, 0],
    }
    
    # 扩展到所有原子
    for key, value in atom_param_dict.items():
        if isinstance(value, list):
            atom_param_dict[key] = [value[0]] * natom
        else:
            atom_param_dict[key] = value
    
    # 随机扰动
    upper = 0.1
    lower = -0.1
    for key, value in atom_param_dict.items():
        if isinstance(value, list):
            if all(isinstance(i, list) for i in value):
                atom_param_dict[key] = [[random.uniform(lower, upper) for _ in range(len(sublist))] 
                                      for sublist in value]
            else:
                atom_param_dict[key] = [random.uniform(lower, upper) for _ in range(len(value))]
        else:
            atom_param_dict[key] = random.uniform(lower, upper)
    
    # 转换为tensor并设置requires_grad
    for key, value in atom_param_dict.items():
        atom_param_dict[key] = torch.tensor(value, dtype=torch.double, requires_grad=True)
    
    return atom_param_dict

def compute_forces(numbers, positions, charge=-1, spin=1, seed=None):
    """计算单个参数集的力"""
    if seed is not None:
        random.seed(seed)
    
    # 创建扰动参数
    atom_param_dict = create_perturbed_params(numbers)
    
    # 设置参数到计算器
    GFN2_XTB_ATOM = dxtb.GFN2_XTB
    GFN2_XTB_ATOM._per_atom_params_dict = atom_param_dict
    
    # 初始化计算器
    opts = {"verbosity": 0, "per_atom": True}
    calc = dxtb.Calculator(numbers, GFN2_XTB_ATOM, opts=opts, **dd)
    
    # 计算能量和力
    pos = positions.clone().requires_grad_(True)
    energy = calc.energy(pos, chrg=charge, spin=spin)
    
    if torch.isnan(energy).any():
        print(f"Warning: NaN energy with seed {seed}")
        return None, None
    
    # 计算力
    (g,) = torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy), 
                              retain_graph=True, create_graph=True)
    forces = -g
    
    return energy, forces

def run_parallel_test(n_calculations=4):
    """运行并行测试"""
    print(f"Loading molecule...")
    
    # 读取分子
    path = Path(__file__).resolve().parent / "molecules" / "nicotine.xyz"
    numbers, positions = read.read(path, ftype="xyz", **dd)
    
    print(f"Running {n_calculations} parallel calculations...")
    
    # 串行测试
    print("\n=== Serial Execution ===")
    start_time = time.time()
    serial_results = []
    for i in range(n_calculations):
        energy, forces = compute_forces(numbers, positions, seed=i*100)
        if energy is not None:
            serial_results.append((energy.item(), forces.sum().item()))
    serial_time = time.time() - start_time
    
    # 并行测试
    print("\n=== Parallel Execution ===")
    start_time = time.time()
    
    # 启动并行任务
    futures = []
    for i in range(n_calculations):
        future = torch.jit.fork(compute_forces, numbers, positions, -1, 1, i*100)
        futures.append(future)
    
    # 收集结果
    parallel_results = []
    for future in futures:
        energy, forces = torch.jit.wait(future)
        if energy is not None:
            parallel_results.append((energy.item(), forces.sum().item()))
    
    parallel_time = time.time() - start_time
    
    # 结果比较
    print(f"\n=== Results ===")
    print(f"Serial time:   {serial_time:.3f}s")
    print(f"Parallel time: {parallel_time:.3f}s")
    print(f"Speedup:       {serial_time/parallel_time:.2f}x")
    
    # 验证结果一致性（前几个结果）
    print(f"\nResults consistency check:")
    for i in range(min(3, len(serial_results), len(parallel_results))):
        s_energy, s_forces = serial_results[i]
        p_energy, p_forces = parallel_results[i]
        energy_diff = abs(s_energy - p_energy)
        forces_diff = abs(s_forces - p_forces)
        print(f"Calc {i+1}: Energy diff = {energy_diff:.6f}, Forces diff = {forces_diff:.6f}")
    
    return serial_time / parallel_time

if __name__ == "__main__":
    print("=" * 60)
    print("DXTB并行力计算测试")
    print("=" * 60)
    
    import sys

    # 从用户输入获取线程配置
    if len(sys.argv) != 3:
        raise ValueError("Please provide two arguments: num_threads and num_interop_threads")

    num_threads = int(sys.argv[1])
    num_interop_threads = int(sys.argv[2])

    # 设置线程配置
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)

    
    # 测试不同的计算数量
    for n_calc in [10]:
        print(f"\n{'='*20} Testing with {n_calc} calculations {'='*20}")
        try:
            speedup = run_parallel_test(n_calc)
            if speedup > 1.2:
                print(f"✓ 良好加速效果: {speedup:.2f}x")
            else:
                print(f"~ 加速效果有限: {speedup:.2f}x")
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            break