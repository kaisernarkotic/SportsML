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

data = data.drop(columns=['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','GAME_STATUS_TEXT'])
pdata = copy.deepcopy(data) #creates a clone of the dataframe (will be processed)
columns_to_fill = ['PTS_home', 'FG_PCT_home','FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']
pdata.loc[:, columns_to_fill] = 0 #fills it with 0

data = data[data['SEASON']==2021]
data = data[(data['TEAM_ID_home']==1610612737) | (data['TEAM_ID_away']==1610612737)]
frow = data[data['SEASON']==2021].index[-1]
print(frow)
pdata.drop(frow)
cdata = data[::-1] #corrected_data


rounds = 1
total_points = 0
avg_points = 0
total_rbds = 0
avg_rbds = 0
total_asts = 0
avg_asts = 0
for index, row in cdata.iterrows():
  home_team = row.loc['TEAM_ID_home'].values[0]
  if(home_team==1610612737):
    total_points+=row['PTS_home'].values[0]
    avg_points=(total_points)/rounds

    total_rbds+=row['REB_home'].values[0]
    avg_rbds=(total_rbds)/rounds

    total_asts+=row['AST_home'].values[0]
    avg_asts=(total_asts)/rounds

    if(index!=frow):
      pdata.loc[index, 'PTS_home'] = avg_points
      pdata.loc[index, 'REB_home'] = avg_rbds
      pdata.loc[index, 'AST_home'] = avg_asts

  else:
    total_points+=row['PTS_away'].values[0]
    avg_points=(total_points)/rounds

    total_rbds+=row['REB_away'].values[0]
    avg_rbds=(total_rbds)/rounds

    total_asts+=row['AST_away'].values[0]
    avg_asts=(total_asts)/rounds

    if(index!=frow):
      pdata.loc[index, 'PTS_away'] = avg_points
      pdata.loc[index, 'AST_away'] = avg_asts
      pdata.loc[index, 'REB_away'] = avg_rbds

  print("average points:" + str(avg_points))
  print("average rebounds:" + str(avg_rbds))
  print("average assists: " + str(avg_asts))
  rounds+=1

print(pdata[pdata['SEASON']==2021])