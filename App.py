
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
pd.options.mode.chained_assignment = None  # default='warn'
from plots import *

from preprocessing import *

from pipeline import *
df_merged = transform_events(num_games=1000)
df_merged['match'] = df_merged['home_team'] + "-" + df_merged['away_team']
possible_matches = df_merged['match'].unique()

model = xgb.XGBClassifier()
model.load_model('test.json')

## data loaded

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)


table = dbc.Table(id='tab', bordered=True)

graph = dcc.Graph(id='graph')
dropdown = dbc.Select(
    id="drop",
    options=[
        {"label": (possible_matches[i]), "value": possible_matches[i]}
        for i in range(len(possible_matches))
    ],
)

checklist = dbc.Select(id = 'check', options = [
    {'label': 'show all', 'value': 'show_all'},
    {'label': 'only home team', 'value': 'home'},
    {'label': 'only away team', 'value': 'away'},],
)

show_a = dcc.Checklist(id = 'checklist',
    options = [
        {'label': 'Show all games', 'value': 'show_all'},]
)

tab1 = dbc.Card([dbc.Tab(dbc.Tab(table, label="Scatter", tab_id="scatter"))])
tab2 = dbc.Card([dropdown, checklist, show_a, dbc.Tab(dbc.Tab(graph, label="2", tab_id="2"))])

app.layout = dbc.Container(html.Div(children=[
    html.H1(children='xG Dashboard'),
    dbc.Row([dbc.Col(tab1), dbc.Col(tab2)])]
    )
)

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Output(component_id = 'tab', component_property='children'),
    Input(component_id='drop', component_property='value'),
    Input(component_id='check', component_property='value'),
    Input(component_id='checklist', component_property='value')
)
def update_graph(dropdown_value, checklist_value, show_all):
    if type(show_all) == list:
        if show_all == []:
            show_all = False
        else:
            show_all = True
    else:
        show_all = False
    
    if dropdown_value == None:
        dropdown_value = possible_matches[0]
    if checklist_value == None:
        checklist_value = 'show_all'
    
    if not show_all:
        df_first = df_merged[df_merged['match'] == dropdown_value]
    else:
        df_first = df_merged
    
    df = df_first[['posx', 'posy', 'goal', 'side', 'minutes', 'quarter', 'distance', 'angle', 'header', 'penalty', 'shot', 'red_card_home_cum',
       'red_card_away_cum', 'yellow_card_home_cum', 'yellow_card_away_cum',
       'attacks_home_cum', 'attacks_away_cum', 'dangerous_attacks_home_cum',
       'dangerous_attacks_away_cum', 'turnover_cum', ]]
    df['side'] = pd.to_numeric(df['side'].apply(lambda x: 1 if x=='away' else 0))
    X = df.drop('goal', axis=1, inplace=False)
    xg_pred = model.predict_proba(X)
    df_first['xG'] = xg_pred[:,1]

    tab = create_table(df_first)
    if checklist_value=='home' or checklist_value=='away':
        df_first = df_first[df_merged['side']==checklist_value]

    return xg_scatter(df_first), tab

def create_table(df):
    xg_team1 = df[df['side']=="home"]['xG'].sum()
    xg_team2 = df[df['side']=="away"]['xG'].sum()
    # round xg_team1 and xg_team2 to 2 decimals
    xg_team1 = '%.2f'%xg_team1
    xg_team2 = '%.2f'%xg_team2
    goals_team1 = df['matchscore'].iloc[-1][0]
    goals_team2 = df['matchscore'].iloc[-1][2]
    
    df = pd.DataFrame({'': ['xG', 'Goals', 'Shots', 'Shots on target', 'Shots off target', 'Shots blocked'],
                    df['home_team'].values[0]: [xg_team1, goals_team1, df['home_shots'].values[0], df['home_shots_target'].values[0],
                        df['home_shots_off_target'].values[0], df['home_shots_blocked'].values[0]],
                    df['away_team'].values[1]: [xg_team2, goals_team2, df['away_shots'].values[0],
                    df['away_shots_target'].values[0], df['away_shots_off_target'].values[0],
                    df['away_shots_blocked'].values[0]]}, index=['xG', 'Goals', 'shots', 'on', 'off', 'blocked'])

    table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
    return table

if __name__ == '__main__':
    app.run_server(debug=True)
