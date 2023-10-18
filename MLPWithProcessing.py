import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import copy

# get data
data = pd.read_csv('games.csv')
#data = data[::-1] #reverses the order of the dataframe

data.columns = pd.MultiIndex.from_tuples([col, ''] for col in data.columns)

data = data.drop(columns=['GAME_DATE_EST','GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','GAME_STATUS_TEXT'])
pdata = copy.deepcopy(data) #creates a clone of the dataframe (will be processed)
columns_to_fill = ['PTS_home', 'FG_PCT_home','FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']
pdata.loc[:, columns_to_fill] = 0 #fills it with 0

data = data[data['SEASON']==2021]

cdata = data[::-1] #corrected_data

rows_to_drop = []

for team_id in range(1610612737, 1610612767):
  fhrow = cdata[cdata['TEAM_ID_home']==team_id].index[0]
  farow = cdata[cdata['TEAM_ID_away']==team_id].index[0]
  frow = max(fhrow,farow)
  rows_to_drop.append(frow)
  rounds = 1
  total_points = 0
  avg_points = 0
  total_rbds = 0
  avg_rbds = 0
  total_asts = 0
  avg_asts = 0
  total_fg = 0
  avg_fg = 0
  total_ft = 0
  avg_ft = 0
  total_fg3 = 0
  avg_fg3 = 0
  for index, row in cdata.iterrows():
    home_team = row.loc['TEAM_ID_home'].values[0]
    away_team = row.loc['TEAM_ID_away'].values[0]
    if(home_team==team_id):
      total_points+=row['PTS_home'].values[0]
      avg_points=(total_points)/rounds

      total_rbds+=row['REB_home'].values[0]
      avg_rbds=(total_rbds)/rounds

      total_asts+=row['AST_home'].values[0]
      avg_asts=(total_asts)/rounds

      total_fg+=row['FG_PCT_home'].values[0]
      avg_fg=(total_fg)/rounds

      total_ft+=row['FT_PCT_home'].values[0]
      avg_ft=(total_ft)/rounds

      total_fg3+=row['FG3_PCT_home'].values[0]
      avg_fg3=(total_fg3)/rounds

      if(index!=frow):
        pdata.iloc[index, 2] = avg_points #'PTS_home' index in second column
        pdata.iloc[index, 7] = avg_rbds #'REB_home' index in second column
        pdata.iloc[index, 6] = avg_asts #'AST_home' index in second column
        pdata.iloc[index, 3] = avg_fg #'FG_PCT_home' index in second column
        pdata.iloc[index, 4] = avg_ft #'FT_PCT_home' index in second column
        pdata.iloc[index, 5] = avg_fg3 #'FG3_PCT_home' index in second column

      rounds+=1

    elif(away_team==team_id):
      total_points+=row['PTS_away'].values[0]
      avg_points=(total_points)/rounds

      total_rbds+=row['REB_away'].values[0]
      avg_rbds=(total_rbds)/rounds

      total_asts+=row['AST_away'].values[0]
      avg_asts=(total_asts)/rounds

      total_fg+=row['FG_PCT_away'].values[0]
      avg_fg=(total_fg)/rounds

      total_ft+=row['FT_PCT_away'].values[0]
      avg_ft=(total_ft)/rounds

      total_fg3+=row['FG3_PCT_away'].values[0]
      avg_fg3=(total_fg3)/rounds

      if(index!=frow):
        pdata.iloc[index, 9] = avg_points #'PTS_away' index in second column
        pdata.iloc[index, 14] = avg_rbds #'REB_away' index in second column
        pdata.iloc[index, 13] = avg_asts #'AST_away' index in second column
        pdata.iloc[index, 10] = avg_fg #'FG_PCT_away' index in second column
        pdata.iloc[index, 11] = avg_ft #'FT_PCT_away' index in second column
        pdata.iloc[index, 12] = avg_fg3 #'FG3_PCT_away' index in second column

      rounds+=1
    #print("average points:" + str(avg_points))
    #print("average rebounds:" + str(avg_rbds))
    #print("average assists: " + str(avg_asts))

for column in pdata.columns:
  pdata[column] = pdata[column].astype(float)

pdata = pdata.drop(rows_to_drop, axis=0)
#print(pdata[pdata['SEASON']==2021])
pdata = pdata[pdata['SEASON']==2021]
pdata = pdata.drop(columns=['TEAM_ID_home', 'TEAM_ID_away'])

inputs = pdata.drop(columns=['HOME_TEAM_WINS'])
labels = pdata['HOME_TEAM_WINS']

input_train, input_test, label_train, label_test = train_test_split(inputs, labels, test_size = 0.25, random_state=42)
batch_size = 64

# convert to torch tensors
input_train = th.tensor(input_train.values, dtype=th.float32)
input_test = th.tensor(input_test.values, dtype=th.float32)
label_train = th.tensor(label_train.values, dtype=th.float32)
label_test = th.tensor(label_test.values, dtype=th.float32)

#print(input_train)

# training and testing data
train_data = TensorDataset(input_train, label_train)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(input_test, label_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self, input, h1, h2, output):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

input_dim = 13
h1_dim = 64
h2_dim = 128
output_dim = 1

model = NeuralNetwork(input_dim, h1_dim, h2_dim, output_dim)

loss_fn = nn.BCELoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

# training
for epoch in range(num_epochs):
  for inputs, labels in train_dataloader:
    #labels = labels.to(th.long)
    #print("inputs: ")
    #print(inputs)
    optimizer.zero_grad()
    pred = model(inputs)
    #print("labels: ")
    #print(labels)
    #print("prediction: ")
    #print(pred)
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