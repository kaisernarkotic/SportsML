import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import copy

# get data
data = pd.read_csv('processedgames.csv')
#data = data[::-1] #reverses the order of the dataframe

data.columns = pd.MultiIndex.from_tuples([col, ''] for col in data.columns)

data = data.drop(columns=['SEASON'])
data.drop(0, inplace=True)

print(data)

inputs = data.drop(columns=['HOME_TEAM_WINS'])
labels = data['HOME_TEAM_WINS']

input_train, input_test, label_train, label_test = train_test_split(inputs, labels, test_size = 0.25, random_state=42) #producing nan values in the label_train variable somehow
batch_size = 64

# convert to torch tensors
input_train = th.tensor(input_train.values, dtype=th.float32)
input_test = th.tensor(input_test.values, dtype=th.float32)
label_train = th.tensor(label_train.values, dtype=th.float32)
label_test = th.tensor(label_test.values, dtype=th.float32)

#print("number of nans: " +str(data["HOME_TEAM_WINS"].isna().sum()))

# training and testing data
train_data = TensorDataset(input_train, label_train)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(input_test, label_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self, input, h1, h2,output):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

input_dim = 14
h1_dim = 64
h2_dim = 64
output_dim = 1

model = NeuralNetwork(input_dim, h1_dim, h2_dim, output_dim)

loss_fn = nn.BCELoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

# training
for epoch in range(num_epochs):
  for inputs, labels in train_dataloader:
    #labels = labels.to(th.long)
    optimizer.zero_grad()
    pred = model(inputs)
    print(pred.max())
    print(labels.max())
    loss = loss_fn(pred.view(-1), labels)
    loss.backward()
    optimizer.step()

#1 means home team won
#column 2 will represent home team wins and column 1 will represent away team wins
# prediction
total = 0
correct = 0

with th.no_grad():
    for inputs, labels in test_dataloader:
        labels = labels.to(th.long)
        correctedpredictions = th.empty(labels.size())
        outputs = model(inputs)
        predicted = (outputs >=0.5).float()
        total += labels.size(0)
        for row in range(predicted.size(0)):
          if(predicted[row]>0.5):
            correctedpredictions[row] = 1
          else:
            correctedpredictions[row] = 0
        correct += (correctedpredictions == labels).sum().item()

print(f'Accuracy of the network on the tests: {100 * correct // total}%')