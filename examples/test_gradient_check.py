#!/usr/bin/env python3
"""
Gradient verification script for per-atom shell scaling in thirdorder.py
"""

import torch

def test_gradient_preservation():
    """Test if our tensor operations preserve gradients properly"""
    
    # Simulate our operations
    n_atoms = 3
    n_shells_total = 6  # Example: 2 shells per atom
    
    # Create test tensors with gradients
    delta_s = torch.randn(n_atoms, requires_grad=True)
    delta_p = torch.randn(n_atoms, requires_grad=True) 
    delta_d = torch.randn(n_atoms, requires_grad=True)
    
    # Simulate concatenation (as in shell_scale_peratom)
    shell_scale_peratom = torch.cat([delta_s, delta_p, delta_d], dim=0)
    print(f"shell_scale_peratom requires_grad: {shell_scale_peratom.requires_grad}")
    
    # Extract components (as in our code)
    delta_s_extracted = shell_scale_peratom[:n_atoms]
    delta_p_extracted = shell_scale_peratom[n_atoms:2*n_atoms]
    delta_d_extracted = shell_scale_peratom[2*n_atoms:3*n_atoms]
    
    print(f"delta_s_extracted requires_grad: {delta_s_extracted.requires_grad}")
    print(f"delta_p_extracted requires_grad: {delta_p_extracted.requires_grad}")
    print(f"delta_d_extracted requires_grad: {delta_d_extracted.requires_grad}")
    
    # Simulate spread_atom_to_shell operation
    # Each atom's value is replicated to its shells
    shells_per_atom = [2, 2, 2]  # 2 shells per atom
    
    def spread_atom_to_shell(atom_values):
        """Simulate ihelp.spread_atom_to_shell"""
        result = []
        for i, n_shell in enumerate(shells_per_atom):
            for _ in range(n_shell):
                result.append(atom_values[i])
        return torch.stack(result)
    
    delta_s_spread = spread_atom_to_shell(delta_s_extracted)
    delta_p_spread = spread_atom_to_shell(delta_p_extracted)
    delta_d_spread = spread_atom_to_shell(delta_d_extracted)
    
    print(f"delta_s_spread requires_grad: {delta_s_spread.requires_grad}")
    
    # Simulate angular momentum masks
    unique_angular = torch.tensor([0, 1, 0, 1, 0, 2])  # s,p,s,p,s,d for 6 shells
    is_s = (unique_angular == 0)
    is_p = (unique_angular == 1)
    is_d = (unique_angular == 2)
    
    # Our final operation
    delta_scale = (delta_s_spread * is_s.float() + 
                   delta_p_spread * is_p.float() + 
                   delta_d_spread * is_d.float())
    
    print(f"delta_scale requires_grad: {delta_scale.requires_grad}")
    
    # Test gradient flow
    loss = delta_scale.sum()
    loss.backward()
    
    print(f"delta_s.grad: {delta_s.grad}")
    print(f"delta_p.grad: {delta_p.grad}")  
    print(f"delta_d.grad: {delta_d.grad}")
    
    # Check if gradients are non-zero (indicates proper flow)
    assert delta_s.grad is not None and not torch.allclose(delta_s.grad, torch.zeros_like(delta_s.grad))
    assert delta_p.grad is not None and not torch.allclose(delta_p.grad, torch.zeros_like(delta_p.grad))
    assert delta_d.grad is not None and not torch.allclose(delta_d.grad, torch.zeros_like(delta_d.grad))
    
    print("✅ Gradient preservation test PASSED!")

if __name__ == "__main__":
    test_gradient_preservation() 