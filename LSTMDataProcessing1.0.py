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
pdata.insert(8, 'MARGIN_home',0)
pdata.insert(16, 'MARGIN_away', 0)
#print(cdata)

rows_to_drop = []

input_dim = 14
h1_dim = 64
h2_dim = 64
output_dim = 1


for team_id in range(1610612737, 1610612767): #for all normal features
  fhrow = cdata[cdata['TEAM_ID_home']==team_id].index[0]
  farow = cdata[cdata['TEAM_ID_away']==team_id].index[0]
  frow = max(fhrow,farow)
  rows_to_drop.append(frow)
  rounds = 1
  pts = []
  rbds = []
  asts = []
  fg = []
  ft = []
  fg3 = []
  for index, row in cdata.iterrows():
    home_team = row.loc['TEAM_ID_home'].values[0]
    away_team = row.loc['TEAM_ID_away'].values[0]
    if(home_team==team_id):
      pts.append(row['PTS_home'].values[0])

      rbds.append(row['REB_home'].values[0])

      asts.append(row['AST_home'].values[0])

      fg.append(row['FG_PCT_home'].values[0])

      ft.append(row['FT_PCT_home'].values[0])

      fg3.append(row['FG3_PCT_home'].values[0])

      if(index!=frow):
        pdata.at[index, 'PTS_home'] = str(pts)
        pdata.at[index, 'REB_home'] = str(rbds) 
        pdata.at[index, 'AST_home'] = str(asts) 
        pdata.at[index, 'FG_PCT_home'] = str(fg) 
        pdata.at[index, 'FT_PCT_home'] = str(ft)
        pdata.at[index, 'FG3_PCT_home'] = str(fg3)

    elif(away_team==team_id):
      pts.append(row['PTS_away'].values[0])

      rbds.append(row['REB_away'].values[0])

      asts.append(row['AST_away'].values[0])

      fg.append(row['FG_PCT_away'].values[0])

      ft.append(row['FT_PCT_away'].values[0])

      fg3.append(row['FG3_PCT_away'].values[0])

      if(index!=frow):
        pdata.at[index, 'PTS_away'] = str(pts)
        pdata.at[index, 'REB_away'] = str(rbds) 
        pdata.at[index, 'AST_away'] = str(asts) 
        pdata.at[index, 'FG_PCT_away'] = str(fg) 
        pdata.at[index, 'FT_PCT_away'] = str(ft)
        pdata.at[index, 'FG3_PCT_away'] = str(fg3)



for team_id in range(1610612737, 1610612767): #again for margin
  fhrow = cdata[cdata['TEAM_ID_home']==team_id].index[0]
  farow = cdata[cdata['TEAM_ID_away']==team_id].index[0]
  frow = max(fhrow,farow)

  margin = []
  for index, row in cdata.iterrows():
    home_team = row.loc['TEAM_ID_home'].values[0]
    away_team = row.loc['TEAM_ID_away'].values[0]
    if(home_team==team_id):
      current_margin+=(row['PTS_home'].values[0]-row['PTS_away'].values[0])
      margin.append(current_margin)
      if(index!=frow):
        pdata.at[index, 'MARGIN_home'] = str(margin)

    if(away_team==team_id):
      current_margin=(row['PTS_away'].values[0]-row['PTS_home'].values[0])
      margin.append(current_margin)
      if(index!=frow):
        pdata.at[index, 'MARGIN_away'] = str(margin)

pdata = pdata.drop(rows_to_drop, axis=0)
pdata = pdata[pdata['SEASON']==2021]
print(pdata)
print("Number of rows(games): " + str(pdata.shape[0]))
pdata = pdata.drop(columns=['TEAM_ID_home', 'TEAM_ID_away'])
pdata.to_csv('lstmprocessedgames.csv', index=False, header=True)