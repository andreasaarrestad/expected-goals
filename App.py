from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import pandas as pd
import xgboost as xgb
import os
pd.options.mode.chained_assignment = None  # default='warn'
from plots import *
from preprocessing import *
from pipeline import *

# Loading event data
df_merged = transform_events(num_games=50)

# Adding match column to fasciliatate dropdown meny where you can choose match
df_merged['match'] = df_merged['home_team'] + "-" + df_merged['away_team']

# All possible matches in df_merged
possible_matches = df_merged['match'].unique()

# Loading models
model = xgb.XGBClassifier()
model.load_model('model.txt')


# Creating dash app
app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# The left-hand side table
table = dbc.Table(id='tab', bordered=True)

# The graph to the right
graph = dcc.Graph(id='graph')

# The choose match dropdown menu
dropdown = dbc.Select(
    id="drop",
    options=[
        {"label": (possible_matches[i]), "value": possible_matches[i]}
        for i in range(len(possible_matches))
    ],
)

# Choose all shots, only home team and only away team
checklist = dbc.Select(id = 'check', options = [
    {'label': 'show all', 'value': 'show_all'},
    {'label': 'only home team', 'value': 'home'},
    {'label': 'only away team', 'value': 'away'},],
)

# Enables displaying all shots in the dataset
show_a = dcc.Checklist(id = 'checklist',
    options = [
        {'label': 'Show all games', 'value': 'show_all'},]
)

# The left hand side table
tab1 = dbc.Card([dbc.Tab(dbc.Tab(table, label="Scatter", tab_id="scatter"))])

# The right hand side elements
tab2 = dbc.Card([dropdown, checklist, show_a, dbc.Tab(dbc.Tab(graph, label="2", tab_id="2"))])

# The layout of the app
app.layout = dbc.Container(html.Div(children=[
    html.H1(children='xG Dashboard'),
    dbc.Row([dbc.Col(tab1), dbc.Col(tab2)])]
    )
)

# A callback function that updates the page based on the dropdown and checklists
@app.callback(
    Output(component_id='graph', component_property='figure'),
    Output(component_id = 'tab', component_property='children'),
    Input(component_id='drop', component_property='value'),
    Input(component_id='check', component_property='value'),
    Input(component_id='checklist', component_property='value')
)
def update_graph(dropdown_value, checklist_value, show_all):
    # Logic for deciding if all matches should be shown
    if type(show_all) == list:
        if show_all == []:
            show_all = False
        else:
            show_all = True
    else:
        show_all = False
    
    # Handling no input case for match
    if dropdown_value == None:
        dropdown_value = possible_matches[0]
    
    # Handling no input case for home/all/away
    if checklist_value == None:
        checklist_value = 'show_all'
    
    # Filtering on match
    if not show_all:
        df_first = df_merged[df_merged['match'] == dropdown_value]
    else:
        df_first = df_merged
    
    # The prediction frame
    df = df_first[['red_card_home_cum', 'red_card_away_cum', 'yellow_card_home_cum', 'yellow_card_away_cum',
        'attacks_home_cum', 'attacks_away_cum', 'dangerous_attacks_home_cum', 'dangerous_attacks_away_cum',
        'quarter', 'distance', 'angle', 'turnover_cum', 'header', 'penalty',#'shot', 
        'preceding_corner', 'preceding_freekick', 'preceding_other', 'preceding_save', 
        'preceding_blocked_shot', 'preceding_dangerous_attack', 'preceding_penalty', 
        'on_target_pred', 'goal_diff', 'goals_up_x_remaining', 'is_home', 'home_lead', 'away_lead', 'goal' 
        ]]

    X = df.drop('goal', axis=1, inplace=False)
    
    # Discarded changes
    # X.loc[X['distance'].isna(), 'distance'] = X['distance'].mean()
    # X.loc[X['angle'].isna(), 'angle'] = X['angle'].mean()

    X.loc[X['distance'].isna(), 'distance'] = 17.243
    X.loc[X['angle'].isna(), 'angle'] = 0.433

    # Predictiing probabilities and adding to dataframe
    xg_pred = model.predict_proba(X)
    df_first['xG'] = xg_pred[:,1]

    # Creating the left hand siden table
    df_tab = df_first[df_first['match'] == dropdown_value]
    tab = create_table(df_tab)
    
    # Filtering on home/away
    if checklist_value=='home' or checklist_value=='away':
        df_first = df_first[df_merged['side']==checklist_value]

    return xg_scatter(df_first), tab

def create_table(df):
    xg_team1 = df[df['side']=="home"]['xG'].sum()
    xg_team2 = df[df['side']=="away"]['xG'].sum()
    
    # Round xg_team1 and xg_team2 to 2 decimals
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
