import pandas as pd
import numpy as np
import os

from preprocessing import encode_shot_types, compute_positional_features, compute_distance_to_goal, compute_angle_to_goal

SEED = 42
EVENTS_PARSER = {"event": ["type", "stime", "side", "mtime", "info", 'posx', 'posy', "matchscore", 'extrainfo']}
TEAM_NAME_PARSER = {'match': ['t1namenatural', 't2namenatural', 't1name', 't2name']}

def get_teams(teams_df):
    home_team, away_team = np.nan, np.nan

    if 't1name' in teams_df.columns:
        home_team = teams_df['t1name'].iloc[0]
    elif 't1namenatural' in teams_df.columns:
        home_team = teams_df['t1namenatural'].iloc[0]
    if 't2name' in teams_df.columns:
        away_team = teams_df['t2name'].iloc[0]
    elif 't2namenatural' in teams_df.columns:
        away_team = teams_df['t2namenatural'].iloc[0]

    return home_team, away_team

def read_xml(dir='./data/'):
    # Iterate over files in dir
    dfs = []
    for i, filename in enumerate(os.listdir(dir)):

        events_df = pd.read_xml(dir + filename, iterparse=EVENTS_PARSER)
        teams_df = pd.read_xml(dir + filename, iterparse=TEAM_NAME_PARSER)

        home_team, away_team = get_teams(teams_df)
        
        events_df['home_team'] = home_team
        events_df['away_team'] = away_team

        events_df['match_id'] = i

        dfs.append(events_df)

    # Concat and get relevant events
    df = pd.concat(dfs).reset_index()

    return df

def add_cumulative_gamestate(df):
    df = df.copy()

    # red card calculation 
    df.loc[(df['type'] == 50) & (df['side'] == 'home'), 'red_card_home_cum'] = 1
    df.loc[(df['type'] == 50) & (df['side'] == 'away'), 'red_card_away_cum'] = 1
    df[['red_card_home_cum','red_card_away_cum']] = df[['red_card_home_cum','red_card_away_cum']].fillna(0)
    df[['red_card_home_cum', 'red_card_away_cum']] = df.groupby('match_id')[['red_card_home_cum', 'red_card_away_cum']].cumsum()

    # yellow card calculation 
    df.loc[(df['type'] == 40) & (df['side'] == 'home'), 'yellow_card_home_cum'] = 1
    df.loc[(df['type'] == 40) & (df['side'] == 'away'), 'yellow_card_away_cum'] = 1
    df[['yellow_card_home_cum','yellow_card_away_cum']] = df[['yellow_card_home_cum','yellow_card_away_cum']].fillna(0)
    df[['yellow_card_home_cum','yellow_card_away_cum']] = df.groupby('match_id')[['yellow_card_home_cum', 'yellow_card_away_cum']].cumsum()

    # attacks
    df.loc[(df['type'] == 1126) & (df['side'] == 'home'), 'attacks_home_cum'] = 1
    df.loc[(df['type'] == 1126) & (df['side'] == 'away'), 'attacks_away_cum'] = 1
    df[['attacks_home_cum','attacks_away_cum']] = df[['attacks_home_cum','attacks_away_cum']].fillna(0)
    df[['attacks_home_cum','attacks_away_cum']] = df.groupby('match_id')[['attacks_home_cum', 'attacks_away_cum']].cumsum()

    # dangerous attacks
    df.loc[(df['type'] == 1029) & (df['side'] == 'home'), 'dangerous_attacks_home_cum'] = 1
    df.loc[(df['type'] == 1029) & (df['side'] == 'away'), 'dangerous_attacks_away_cum'] = 1
    df[['dangerous_attacks_home_cum','dangerous_attacks_away_cum']] = df[['dangerous_attacks_home_cum','dangerous_attacks_away_cum']].fillna(0)
    df[['dangerous_attacks_home_cum','dangerous_attacks_away_cum']] = df.groupby('match_id')[['dangerous_attacks_home_cum', 'dangerous_attacks_away_cum']].cumsum()

    # turnover 
    df['prev_side'] = df.groupby('match_id')['side'].shift()
    home_loses_possession = (df['prev_side'] == 'home') & (df['side'] == 'away')
    away_loses_possession = (df['side'] == 'home') & (df['prev_side'] == 'away')
    df.loc[home_loses_possession | away_loses_possession, 'turnover_cum'] = 1
    df['turnover_cum'] = df['turnover_cum'].fillna(0)
    df['turnover_cum'] = df.groupby('match_id')['turnover_cum'].cumsum()
    df.drop(['prev_side'], axis=1)

    return df

def transform_events(compute_solid_angle=False, relevant_events={30, 155, 156, 172, 666}):
    # Relevant events defaults to goal, shot on/off target, shot blocked, pentaly missed
    # Compute solid angle is time consuming, optional

    df = read_xml()

    # engineer cumulative game state features
    df = add_cumulative_gamestate(df)

    # Get relevant events only
    df = df[df['type'].isin(relevant_events)]

    # Encode shot types
    df = encode_shot_types(df)

    # Get quarter of the match
    df['minutes'] = df['mtime'].str.replace(r'(\:.*)', '', regex=True).astype(int)
    df['quarter'] = pd.cut(df['minutes'], bins=[0, 15, 30, 45, 60, 75, 120], labels=False, retbins=True, right=False)[0]

    # Distance angle
    df['distance'] = compute_distance_to_goal(df)
    df['angle'] = compute_angle_to_goal(df)
    if compute_solid_angle:
        df['solid_angle'] = df.apply(
            lambda row: compute_positional_features(row['posx'], row['posy'], row['side']), axis=1
        )

    df = pd.concat([df, pd.get_dummies(df['shot_type'])], axis=1)
    return df

    