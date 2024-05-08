import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import copy

# get data
data = th.load("LSTMProcessedData.pt")
#data = data[::-1] #reverses the order of the dataframe

data = pd.DataFrame(data)

def tensor_to_int(tensor):
    if tensor.dim() == 0:  # Check if tensor is 0-dimensional
        return int(tensor.item())  
    else:
        return tensor.tolist()  # Return unchanged if not a tensor

# Iterate over each cell in the DataFrame and convert tensors to integers
for col in data.columns:
    data[col] = data[col].apply(lambda x: [tensor_to_int(tensor) for tensor in x])

labels = data['HOME_TEAM_WINS:']

inputs = data.pop('HOME_TEAM_WINS:')

print(labels)
print(inputs)

input_train, input_test, label_train, label_test = train_test_split(inputs, labels, test_size = 0.25, random_state=42)
batch_size = 64

print(type(input_train))
print(input_test)
print(label_train)
print(label_test)

# convert to torch tensors
input_train = th.tensor(input_train, dtype=th.float32)
input_test = th.tensor(input_test, dtype=th.float32)
label_train = th.tensor(label_train, dtype=th.float32)
label_test = th.tensor(label_test, dtype=th.float32)

# training and testing data
train_data = TensorDataset(input_train, label_train)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(input_test, label_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = th.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = th.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use only the last output in the sequence
        return out


# Define model, loss function, and optimizer
input_size = input_train.size(-1)
hidden_size = 64
num_layers = 1
output_size = label_train.size(-1)


model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for sequences, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(sequences.unsqueeze(-1))  # Add extra dimension for input_size
        loss = criterion(outputs.squeeze(), labels)  # Squeeze outputs to match labels shape
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Make predictions
with th.no_grad():
  for sequences, labels in test_dataloader:
    outputs = model(sequences.unsqueeze(-1))  # Add extra dimension for input_size
    predictions = th.round(th.sigmoid(outputs)).squeeze()  # Squeeze outputs to match labels shape
    accuracy = (predictions == labels).sum().item() / labels.size(0)
    print(f'Test Accuracy: {accuracy:.4f}')


