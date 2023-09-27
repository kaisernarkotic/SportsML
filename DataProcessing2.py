import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# get data
data = pd.read_csv('games.csv')
#data = data[::-1] #reverses the order of the dataframe

data.columns = pd.MultiIndex.from_tuples([col, ''] for col in data.columns)

data = data.drop(columns=['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','GAME_STATUS_TEXT'])
data = data[data['SEASON']==2021]
data = data[(data['TEAM_ID_home']==1610612737) | (data['TEAM_ID_away']==1610612737)]

cdata = data[::-1] #corrected_data
#print(cdata)

rounds = 1
total_points = 0
avg_points = 0
total_rbds = 0
avg_rbds = 0
total_asts = 0
avg_asts = 0
for index, row in cdata.iterrows():
  home_team = row.loc['TEAM_ID_home'].values[0]
  #print(row['PTS_home'].values[0])
  if(home_team==1610612737):
    total_points+=row['PTS_home'].values[0]
    avg_points=(total_points)/rounds
    total_rbds+=row['REB_home'].values[0]
    avg_rbds=(total_rbds)/rounds
    total_asts+=row['AST_home'].values[0]
    avg_asts=(total_asts)/rounds
  else:
    total_points+=row['PTS_away'].values[0]
    avg_points=(total_points)/rounds
    total_rbds+=row['REB_away'].values[0]
    avg_rbds=(total_rbds)/rounds
    total_asts+=row['AST_away'].values[0]
    avg_asts=(total_asts)/rounds
  print("average points:" + str(avg_points))
  print("average rebounds:" + str(avg_rbds))
  print("average assists: " + str(avg_asts))
  rounds+=1