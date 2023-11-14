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

for team_id in range(1610612737, 1610612767): #for all normal features
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
        margin_home = row['PTS_home'].values[0]-row['PTS_away'].values[0]

        pdata.iloc[index, 2] = avg_points #'PTS_home' index in second column
        pdata.iloc[index, 7] = avg_rbds #'REB_home' index in second column
        pdata.iloc[index, 6] = avg_asts #'AST_home' index in second column
        pdata.iloc[index, 3] = avg_fg #'FG_PCT_home' index in second column
        pdata.iloc[index, 4] = avg_ft #'FT_PCT_home' index in second column
        pdata.iloc[index, 5] = avg_fg3 #'FG3_PCT_home' index in second column

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
        pdata.iloc[index, 10] = avg_points #'PTS_away' index in second column
        pdata.iloc[index, 15] = avg_rbds #'REB_away' index in second column
        pdata.iloc[index, 14] = avg_asts #'AST_away' index in second column
        pdata.iloc[index, 11] = avg_fg #'FG_PCT_away' index in second column
        pdata.iloc[index, 12] = avg_ft #'FT_PCT_away' index in second column
        pdata.iloc[index, 13] = avg_fg3 #'FG3_PCT_away' index in second column

    rounds+=1
    #print("average points:" + str(avg_points))
    #print("average rebounds:" + str(avg_rbds))
    #print("average assists: " + str(avg_asts))



for team_id in range(1610612737, 1610612767): #again for margin
  fhrow = cdata[cdata['TEAM_ID_home']==team_id].index[0]
  farow = cdata[cdata['TEAM_ID_away']==team_id].index[0]
  frow = max(fhrow,farow)
  rounds = 1
  total_margin = 0
  avg_margin = 0
  for index, row in cdata.iterrows():
    home_team = row.loc['TEAM_ID_home'].values[0]
    away_team = row.loc['TEAM_ID_away'].values[0]
    if(home_team==team_id):
      total_margin+=(row['PTS_home'].values[0]-row['PTS_away'].values[0])
      avg_margin=(total_margin)/rounds
      if(index!=frow):
        pdata.iloc[index, 8] = avg_margin #'MARGIN_home' index in second column
    if(away_team==team_id):
      total_margin+=(row['PTS_away'].values[0]-row['PTS_home'].values[0])
      avg_margin=(total_margin)/rounds
      if(index!=frow):
        pdata.iloc[index, 16] = avg_margin #'MARGIN_home' index in second column
    rounds+=1


for column in pdata.columns:
  pdata[column] = pdata[column].astype(float)

pdata = pdata.drop(rows_to_drop, axis=0)
pdata = pdata[pdata['SEASON']==2021]
print(pdata)
pdata = pdata.drop(columns=['TEAM_ID_home', 'TEAM_ID_away'])
pdata.to_csv('processedgames.csv', index=False, header=True)