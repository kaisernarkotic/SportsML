import pandas as pd
import torch as th
import numpy as np
import copy
import csv

# get data
data = pd.read_csv('games.csv')
#data = data[::-1] #reverses the order of the dataframe

data.columns = pd.MultiIndex.from_tuples([col, ''] for col in data.columns)

data = data.drop(columns=['GAME_DATE_EST','GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','GAME_STATUS_TEXT'])
#empty_data = pd.DataFrame(index=df.index, columns=df.columns)

data = data[data['SEASON']==2021]

cdata = data[::-1] #corrected data (pandas dataframe)



pdata = copy.deepcopy(data) #creates a clone of the dataframe (will be processed)
columns_to_fill = ['PTS_home', 'FG_PCT_home','FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']
#pdata.loc[:, columns_to_fill] = th.tensor(blank,dtype=th.float32) #fills it with tensors

length = 20
blank = [0] * length
blank_list = []
for i in range(len(pdata)):
  tensor = th.tensor(blank, dtype=th.float32)
  blank_list.append(tensor)
for column in columns_to_fill:
    pdata[column] = blank_list
pdata.insert(8, 'MARGIN_home',0)
pdata.insert(16, 'MARGIN_away', 0)
print(pdata)

rows_to_drop = []

rounds = 0
for team_id in range(1610612737, 1610612767): #for all normal features
  fhrow = cdata[cdata['TEAM_ID_home']==team_id].index[0]
  farow = cdata[cdata['TEAM_ID_away']==team_id].index[0]
  frow = max(fhrow,farow)
  rows_to_drop.append(frow)
  base_pts = [0] * length
  base_rbds = [0] * length
  base_asts = [0] * length
  base_fg = [0] * length
  base_ft = [0] * length
  base_fg3 = [0] * length
  total_margin = 0
  avg_margin = 0
  for index, row in cdata.iterrows():
    home_team = row.loc['TEAM_ID_home'].values[0]
    away_team = row.loc['TEAM_ID_away'].values[0]
    pts = copy.deepcopy(base_pts)
    rbds = copy.deepcopy(base_rbds)
    asts = copy.deepcopy(base_asts)
    fg = copy.deepcopy(base_fg)
    ft = copy.deepcopy(base_ft)
    fg3 = copy.deepcopy(base_fg3)
    if(home_team==team_id):
      if(rounds<length):

        pts[rounds] = row['PTS_home'].values[0]

        rbds[rounds] = row['REB_home'].values[0]

        asts[rounds] = row['AST_home'].values[0]

        fg[rounds] = row['FG_PCT_home'].values[0]

        ft[rounds] = row['FT_PCT_home'].values[0]

        fg3[rounds] = row['FG3_PCT_home'].values[0]
      else:
        pts.pop(0)
        rbds.pop(0)
        asts.pop(0)
        fg.pop(0)
        ft.pop(0)
        fg3.pop(0)

        pts.append(row['PTS_home'].values[0])

        rbds.append(row['REB_home'].values[0])

        asts.append(row['AST_home'].values[0])

        fg.append(row['FG_PCT_home'].values[0])

        ft.append(row['FT_PCT_home'].values[0])

        fg3.append(row['FG3_PCT_home'].values[0])

      if index not in rows_to_drop:
        """total_margin+=(row['PTS_home'].values[0]-row['PTS_away'].values[0])
        avg_margin=(total_margin)/(rounds+1)
        pdata.iloc[index, 8] = avg_margin #'MARGIN_home' index in eighth column"""
        print(th.tensor(pts, dtype=th.float32).size())
        print(pdata.iloc[index, 2].size())
        pdata.iloc[index, 2] = th.tensor(pts, dtype=th.float32) #'PTS_home' index in second column
        pdata.iloc[index, 7] = th.tensor(rbds, dtype=th.float32) #'REB_home' index in seventh column
        pdata.iloc[index, 6] = th.tensor(asts, dtype=th.float32) #'AST_home' index in sixth column
        pdata.iloc[index, 3] = th.tensor(fg, dtype=th.float32) #'FG_PCT_home' index in third column
        pdata.iloc[index, 4] = th.tensor(ft, dtype=th.float32) #'FT_PCT_home' index in fourth column
        pdata.iloc[index, 5] = th.tensor(fg3, dtype=th.float32) #'FG3_PCT_home' index in fifth column

      rounds+=1

    elif(away_team==team_id):
      if(rounds<length):
        pts[rounds] = row['PTS_away'].values[0]

        rbds[rounds] = row['REB_away'].values[0]

        asts[rounds] = row['AST_away'].values[0]

        fg[rounds] = row['FG_PCT_away'].values[0]

        ft[rounds] = row['FT_PCT_away'].values[0]

        fg3[rounds] = row['FG3_PCT_away'].values[0]
      else:
        pts.pop(0)
        rbds.pop(0)
        asts.pop(0)
        fg.pop(0)
        ft.pop(0)
        fg3.pop(0)

        pts.append(row['PTS_away'].values[0])

        rbds.append(row['REB_away'].values[0])

        asts.append(row['AST_away'].values[0])

        fg.append(row['FG_PCT_away'].values[0])

        ft.append(row['FT_PCT_away'].values[0])

        fg3.append(row['FG3_PCT_away'].values[0])
      if index not in rows_to_drop:
        """total_margin+=(row['PTS_away'].values[0]-row['PTS_home'].values[0])
        avg_margin=(total_margin)/(rounds+1)
        pdata.iloc[index, 16] = avg_margin #'MARGIN_away' index in sixteenth column"""
        
        pdata.iloc[index, 10] = th.tensor(pts, dtype=th.float32) #'PTS_away' index in tenth column
        pdata.iloc[index, 15] = th.tensor(rbds, dtype=th.float32) #'REB_away' index in fifteenth column
        pdata.iloc[index, 14] = th.tensor(asts, dtype=th.float32) #'AST_away' index in fourteenth column
        pdata.iloc[index, 11] = th.tensor(fg, dtype=th.float32) #'FG_PCT_away' index in eleventh column
        pdata.iloc[index, 12] = th.tensor(ft, dtype=th.float32) #'FT_PCT_away' index in twelfth column
        pdata.iloc[index, 13] = th.tensor(fg3, dtype=th.float32) #'FG3_PCT_away' index in thirteenth column



      rounds+=1
    base_pts = copy.deepcopy(pts)
    base_rbds = copy.deepcopy(rbds)
    base_asts = copy.deepcopy(asts)
    base_fg = copy.deepcopy(fg)
    base_ft = copy.deepcopy(ft)
    base_fg3 = copy.deepcopy(fg3)


rounds = 0
for team_id in range(1610612737, 1610612767): #again for margin
  fhrow = cdata[cdata['TEAM_ID_home']==team_id].index[0]
  farow = cdata[cdata['TEAM_ID_away']==team_id].index[0]
  frow = max(fhrow,farow)
  base_margin = [0] * length

  for index, row in cdata.iterrows():
    home_team = row.loc['TEAM_ID_home'].values[0]
    away_team = row.loc['TEAM_ID_away'].values[0]
    margin = copy.deepcopy(base_margin)
    if(home_team==team_id):
      current_margin+=(row['PTS_home'].values[0]-row['PTS_away'].values[0])
      if(rounds<length):
        margin[rounds] = current_margin
      else:
        margin.pop(0)
        margin.append(current_margin)
      if index not in rows_to_drop:
        pdata.iloc[index, 8] = th.tensor(margin, dtype=th.float32) #'MARGIN_home' index in eighth column
      rounds+=1

    if(away_team==team_id):
      current_margin=(row['PTS_away'].values[0]-row['PTS_home'].values[0])
      if(rounds<length):
        margin[rounds] = current_margin
      else:
        margin.pop(0)
        margin.append(current_margin)
      if index not in rows_to_drop:
        pdata.iloc[index, 16] = th.tensor(margin, dtype=th.float32) #'MARGIN_away' index in sixteenth column
      rounds+=1
    base_margin = copy.deepcopy(margin)

for index, row in cdata.iterrows(): #set all the home team wins values
  if index not in rows_to_drop:
    pdata['HOME_TEAM_WINS:'].append(th.tensor(row['HOME_TEAM_WINS'].values[0], dtype=th.float32))