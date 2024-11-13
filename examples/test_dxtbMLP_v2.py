import torch
import torch.nn as nn
import torch.optim as optim
import dxtb
from test_e3nn import InvariantPolynomial

# for e3nn
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct

# Define the MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.initialize_parameters()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_parameters(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)



if __name__ == "__main__":
    # test the function of InvariantPolynomial
    num_z = 3
    lmax = 2
    irreps_out = "64x0e + 24x1e + 24x1o + 16x2e + 16x2o" # these mean the output of the model
    model = InvariantPolynomial(irreps_out, num_z, lmax)
    print(model)
    
    # test input 
    data = Data( 
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5], [0.0, 0.0, 3.5]]),
        z=torch.tensor([3, 1, 4, 1]),
        batch=torch.tensor([0, 0, 0, 1]),
    )
    print(data)

    # test the forward function
    output = model(data)
    print(output)
    
    
    
    
    
# if __name__ == "__main__":
    
#     # define config
#     dd = {"dtype": torch.double, "device": torch.device("cpu")}
#     numbers = torch.tensor([3, 1, 4], device=dd["device"]) # atom numbers
#     positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]],
#                             **dd)


#     # Initialize the MLP
#     input_dim = 3  # Number of input features
#     output_dim = 4*3  # Number of output features # show be able to predict for each atom
#     mlp = MLP(input_dim, output_dim).to(dd["device"])

#     # Define the optimizer
#     optimizer = optim.Adam(mlp.parameters(), lr=0.1)

#     # target_energy
#     target_energy = torch.tensor(-1.0, **dd)

#     # Training loop
#     num_epochs = 100
#     loss_list = []
#     for epoch in range(num_epochs):
#         # Forward pass: predict parameters
#         # use atom numbers to predict the parameters
#         input_features = numbers.float()
#         predicted_params = mlp(input_features)

#         # Update tensor_dict with predicted parameters
#         tensor_dict = {}
#         natom = 3

#         zeff = predicted_params[0*natom:1*natom].cpu().detach().numpy()
#         arep = predicted_params[1*natom:2*natom].cpu().detach().numpy()
#         en = predicted_params[2*natom:3*natom].cpu().detach().numpy()
#         gam = predicted_params[3*natom:4*natom].cpu().detach().numpy()
        
#         tensor_dict["zeff"] = torch.tensor(zeff, requires_grad=True, **dd)
#         tensor_dict["arep"] = torch.tensor(arep, requires_grad=True, **dd)
#         tensor_dict["en"] = torch.tensor(en, requires_grad=True, **dd)
#         tensor_dict["gam"] = torch.tensor(gam, requires_grad=True, **dd)
        
#         print(f"predicted_params: {tensor_dict}")

#         # Compute the energy
#         pos = positions.clone().requires_grad_(True)
#         calc = dxtb.calculators.GFN1Calculator(numbers, tensor_dict=tensor_dict, **dd)
#         energy = calc.get_energy(pos)
        
#         if energy is None:
#             print("Energy is None")
#             exit(1)

#         # Compute the loss (mean squared error)
#         loss_xtb = torch.mean((energy - target_energy) ** 2)  # Replace target_energy with your target value

#         # Compute the middle gradient
#         g_zeff = torch.autograd.grad(loss_xtb, tensor_dict["zeff"], retain_graph=True)[0]
#         g_arep = torch.autograd.grad(loss_xtb, tensor_dict["arep"], retain_graph=True)[0]
#         g_en = torch.autograd.grad(loss_xtb, tensor_dict["en"], retain_graph=True)[0]
#         g_gam = torch.autograd.grad(loss_xtb, tensor_dict["gam"], retain_graph=True)[0]

#         # Stack the gradients
#         middle_gradient = torch.stack([g_zeff, g_arep, g_en, g_gam])

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         predicted_params.backward(middle_gradient.view(-1))
#         optimizer.step()

#         calc.reset()

#         # Print the loss
#         if (epoch + 1) % 1 == 0:
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_xtb.item()}")
            
#         loss_list.append((loss_xtb.item(), middle_gradient.tolist()))

#     # After training, you can use the trained MLP to predict parameters for new inputs


#     # save the loss to a file
#     with open("loss.txt", "w") as f:
#         for loss in loss_list:
#             f.write(f"{loss}\n")