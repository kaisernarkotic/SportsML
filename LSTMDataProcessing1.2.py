import pandas as pd
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

pdata = {} #dictionary of lists

pdata['SEASON:'] = []
pdata['PTS_home:'] = []
pdata['FG_PCT_home:'] = []
pdata['FT_PCT_home:'] = []
pdata['FG3_PCT_home:'] = []
pdata['AST_home:'] = []
pdata['REB_home:'] = []
pdata['PTS_away:'] = []
pdata['FG_PCT_away:'] = []
pdata['FT_PCT_away:'] = []
pdata['FG3_PCT_away:'] = []
pdata['AST_away:'] = []
pdata['REB_away:'] = []
pdata['HOME_TEAM_WINS:'] = []
pdata['MARGIN_home:'] = []
pdata['MARGIN_away:'] = []

for key in pdata:
    print(key)

rows_to_drop = []

rounds = 0
length = 20


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
      if(rounds<20):

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
        pdata['PTS_home:'].append(pts)
        pdata['REB_home:'].append(rbds)
        pdata['AST_home:'].append(asts)
        pdata['FG_PCT_home:'].append(fg)
        pdata['FT_PCT_home:'].append(ft)
        pdata['FG3_PCT_home:'].append(fg3)
      rounds+=1

    elif(away_team==team_id):
      if(rounds<20):
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
        pdata['PTS_away:'].append(pts)
        pdata['REB_away:'].append(rbds)
        pdata['AST_away:'].append(asts)
        pdata['FG_PCT_away:'].append(fg)
        pdata['FT_PCT_away:'].append(ft)
        pdata['FG3_PCT_away:'].append(fg3)
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
      if(rounds<20):
        margin[rounds] = current_margin
      else:
        margin.pop(0)
        margin.append(current_margin)
      if index not in rows_to_drop:
        pdata['MARGIN_home:'].append(margin)
      rounds+=1

    if(away_team==team_id):
      current_margin=(row['PTS_away'].values[0]-row['PTS_home'].values[0])
      if(rounds<20):
        margin[rounds] = current_margin
      else:
        margin.pop(0)
        margin.append(current_margin)
      if(index not in rows_to_drop):
        pdata['MARGIN_away:'].append(margin)
      rounds+=1
    base_margin = copy.deepcopy(margin)


filename = 'lstmprocessedgames.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(pdata.keys())  # Write column headers
    for row in zip(*pdata.values()):
      writer.writerows(row)  # Write rows