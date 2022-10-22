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

def get_shots(dir, filename):
    shotsontarget = pd.read_xml(dir+filename, iterparse={'shotsontarget': ['t1', 't2']})
    shotsoftarget = pd.read_xml(dir+filename, iterparse={'shotsofftarget': ['t1', 't2']})
    shotsblocked = pd.read_xml(dir+filename, iterparse={'shotsblocked': ['t1', 't2']})

    home_team = [shotsontarget.t1.values[0], shotsoftarget.t1.values[0], shotsblocked.t1.values[0]]
    away_team = [shotsontarget.t2.values[0], shotsoftarget.t2.values[0], shotsblocked.t2.values[0]]

    return home_team, away_team



def read_xml(dir='./startcode/', num_games = False):
    # Iterate over files in dir
    dfs = []
    if not num_games:
        num_games = len(os.listdir(dir))
    
    for filename in os.listdir(dir)[:num_games]:

        events_df = pd.read_xml(dir + filename, iterparse=EVENTS_PARSER)
        teams_df = pd.read_xml(dir + filename, iterparse=TEAM_NAME_PARSER)

        home_team, away_team = get_teams(teams_df)
        
        events_df['home_team'] = home_team
        events_df['away_team'] = away_team

        shots = get_shots(dir, filename)
        events_df['home_shots_target'] = shots[0][0]
        events_df['away_shots_target'] = shots[1][0]
        events_df['home_shots_off_target'] = shots[0][1]
        events_df['away_shots_off_target'] = shots[1][1]
        events_df['home_shots_blocked'] = shots[0][2]
        events_df['away_shots_blocked'] = shots[1][2]
        events_df['home_shots'] = events_df['home_shots_target'] + events_df['home_shots_off_target'] + events_df['home_shots_blocked']
        events_df['away_shots'] = events_df['away_shots_target'] + events_df['away_shots_off_target'] + events_df['away_shots_blocked']
        
        dfs.append(events_df)

    # Concat and get relevant events
    df = pd.concat(dfs).reset_index()

    return df

def transform_events(compute_solid_angle=False, relevant_events={30, 155, 156, 172, 666}, num_games = False):
    # Relevant events defaults to goal, shot on/off target, shot blocked, pentaly missed
    # Compute solid angle is time consuming, optional

    df = read_xml(num_games = num_games)

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

    