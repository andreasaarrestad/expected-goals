import pandas as pd
import numpy as np
import os

from event_features import add_on_target_prob, encode_shot_types, compute_positional_features, compute_distance_to_goal, \
                           compute_angle_to_goal, compute_if_shot_is_in_boxes

from game_state_features import add_score_features, add_cumulative_gamestate, \
                                encode_prev_event, get_shots, get_teams

SEED = 42
EVENTS_PARSER = {"event": ["type", "stime", "side", "mtime", "info", 'posx', 'posy', "matchscore", 'extrainfo']}
TEAM_NAME_PARSER = {'match': ['t1namenatural', 't2namenatural', 't1name', 't2name']}


def read_xml(dir='./data/', num_games = False):
    # Iterate over files in dir
    dfs = []

    if num_games is False:
        num_games = len(os.listdir(dir))
    
    for i, filename in enumerate(os.listdir(dir)):
        if i>=num_games:
            break

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
        events_df['match_id'] = i

        events_df['weather_condition'] = events_df.loc[events_df['type'] == 164, 'extrainfo'].iloc[-1] if (events_df['type'] == 164).sum() else 0
        events_df['pitch_condition'] = events_df.loc[events_df['type'] == 1014, 'extrainfo'].iloc[-1] if (events_df['type'] == 1014).sum() else 0


        dfs.append(events_df)

    # Concat and get relevant events
    df = pd.concat(dfs).reset_index(drop=True)

    return df


def transform_events(compute_solid_angle=False, relevant_events={30, 155, 156, 172, 666}, num_games=False):
    # Relevant events defaults to goal, shot on/off target, shot blocked, pentaly missed
    # Compute solid angle is time consuming, optional

    df = read_xml(num_games = num_games)

    # Goals.
    df.loc[df['type'] == 30, 'goal'] = 1
    df.loc[df['type'] != 30, 'goal'] = 0

    # Get quarter of the match
    df['minutes'] = df['mtime'].str.replace(r'(\:.*)', '', regex=True).astype(int)
    df['quarter'] = pd.cut(df['minutes'], bins=[0, 15, 30, 45, 60, 75, 120], labels=False, retbins=True, right=False)[0]
    df['quarter'] = df['quarter'].fillna(0)

    # engineer cumulative game state features
    df = add_cumulative_gamestate(df)

    df = encode_prev_event(df)

    df['is_home'] = (df['side'] == 'home').astype(int)

    # Get relevant events only
    df = df[df['type'].isin(relevant_events)]

    # Encode shot types
    df = encode_shot_types(df)


    # compute if the shot is done within one of the boxes
    df = compute_if_shot_is_in_boxes(df)


    # Distance angle
    df['distance'] = compute_distance_to_goal(df)
    df['angle'] = compute_angle_to_goal(df, between_goal_posts=True)
    df['angle_over_distance'] = np.abs(df['angle']).div(df['distance'].replace(0, 0.01), axis=0)
    
    df.loc[df['header'] == 1, 'distance'] = df['distance']**2

    df = add_score_features(df)
    df = add_on_target_prob(df, modelchoice='logit')

    if compute_solid_angle:
        df['solid_angle'] = df.apply(
            lambda row: compute_positional_features(row['posx'], row['posy'], row['side'], row['header']), axis=1
        )

    return df

    