import tensorflow_datasets as tfds

# Load the QM9 dataset
dataset, info = tfds.load('qm9', with_info=True, as_supervised=True, data_dir='./tfds_data')
train_dataset, test_dataset = dataset['train'], dataset['test']


import torch
from torch.utils.data import DataLoader, Dataset


class TFDataLoader(Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = list(tf_dataset)

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        image, label = self.tf_dataset[idx]
        return image.numpy(), label.numpy()

# Convert the TensorFlow dataset to PyTorch DataLoader
train_loader = DataLoader(TFDataLoader(train_dataset), batch_size=32, shuffle=True)
test_loader = DataLoader(TFDataLoader(test_dataset), batch_size=32, shuffle=False)



import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
