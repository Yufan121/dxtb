import tensorflow_datasets as tfds
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Load the QM9 dataset
dataset, info = tfds.load('qm9', with_info=True, data_dir='./tfds_data')

# Print a dataset briefing
print("Dataset Name:", info.name)
print("Number of Examples:", info.splits['train'].num_examples)
print("Features:", info.features)
print("Description:", info.description)
print("Citation:", info.citation)


# Custom Dataset class to convert TensorFlow dataset to PyTorch DataLoader
class TFDataLoader(Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = list(tfds.as_numpy(tf_dataset))

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        example = self.tf_dataset[idx]
        # Extract features and labels from the example
        features = example['positions']  # Adjust based on the actual key in QM9 dataset
        label = example['energy']  # Adjust based on the actual key in QM9 dataset
        return features, label

# Convert the TensorFlow dataset to PyTorch DataLoader
train_loader = DataLoader(TFDataLoader(dataset['train']), batch_size=32, shuffle=True)
test_loader = DataLoader(TFDataLoader(dataset['test']), batch_size=32, shuffle=False)

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(29 * 3, 128)  # Adjust input size based on QM9 dataset
        self.fc2 = nn.Linear(128, 1)  # Adjust output size based on your task

    def forward(self, x):
        x = x.view(-1, 29 * 3)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
criterion = nn.MSELoss()  # Adjust loss function based on your task
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for features, labels in train_loader:
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluation loop
model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        outputs = model(features)
        loss = criterion(outputs, labels)
        print(f'Test Loss: {loss.item()}')
