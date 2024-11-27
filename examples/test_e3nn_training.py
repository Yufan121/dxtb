import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from e3nn import o3
from test_e3nn import InvariantPolynomial, compute_local_environment

if __name__ == "__main__":
    
    # Define the model
    model = InvariantPolynomial("0e+0o", num_z=3, lmax=3)

    # Ensure all model parameters are of type torch.float64
    model = model.double()
    # print input/output features
    # print(f'num_input_features: {model.num_input_features}, num_output_features: {model.num_output_features}')

    # Define a simple loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate toy data
    torch.set_default_dtype(torch.float64)



    # Example usage
    pos1 = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.5],
        ],
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 2.0],
        ]]
    )

    z = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 1]])
    relative_pos, neighbor_type = compute_local_environment(pos1, z=z, radius=1000) # into shape (F, N, N-1, 3) and (F, N, N-1, 1)
    print(f'relative_pos.shape: {relative_pos.shape}, neighbor_type.shape: {neighbor_type.shape}')
    # print(f'relative_pos: {relative_pos}, neighbor_type: {neighbor_type}')

    
    
    # reshape relative_pos and neighbor_type for training (dim 2)
    # make shape (F, N, N-1, 3) -> (F*N, N-1, 3), (F, N, N-1, 1) -> (F*N, N-1)
    relative_pos = relative_pos.view(-1, relative_pos.size(2), relative_pos.size(3))
    neighbor_type = neighbor_type.view(-1, neighbor_type.size(2))
    
    # Generate random targets for each training example
    targets = torch.randn(relative_pos.size(0), 2)

    # reduce the first dim, make list
    dataset = [Data(pos=pos, z=z) for pos, z in zip(relative_pos, neighbor_type)]
    print(f'dataset: {dataset}')
    
    # Create a dataset and a data loader
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    

    
    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            
            # Use the corresponding targets
            target = targets[i * data_loader.batch_size:(i + 1) * data_loader.batch_size]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    print("Training complete!")

    # Save the model
    torch.save(model.state_dict(), 'toy_test.model.pth')
