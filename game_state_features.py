import numpy as np
import pandas as pd



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
    # try statements to cover ParserError if xml does not have the right tag
    try:
        shotsontarget = pd.read_xml(dir+filename, iterparse={'shotsontarget': ['t1', 't2']})
    except:
        shotsontarget = pd.DataFrame(columns=['t1', 't2'], data=[[0, 0]])
    try:
        shotsoftarget = pd.read_xml(dir+filename, iterparse={'shotsofftarget': ['t1', 't2']})
    except:
        shotsoftarget = pd.DataFrame(columns=['t1', 't2'], data=[[0, 0]])
    try:
        shotsblocked = pd.read_xml(dir+filename, iterparse={'shotsblocked': ['t1', 't2']})
    except:
        shotsblocked = pd.DataFrame(columns=['t1', 't2'], data=[[0, 0]])
    home_team = [shotsontarget.t1.values[0], shotsoftarget.t1.values[0], shotsblocked.t1.values[0]]
    away_team = [shotsontarget.t2.values[0], shotsoftarget.t2.values[0], shotsblocked.t2.values[0]]
    return home_team, away_team

# Events considered: free kick, corner, save, penalty, blocked shot, save, dangerous attack
def find_prev_event(prev_events):
    events = [150, 154, 157, 161, 172, 1029]
    for event in prev_events.values:
        if event in events:
            return event
    return np.nan

def encode_prev_event(df: pd.DataFrame):
    df['prev_type'] = df['type'][::-1].rolling(6, closed='left').apply(find_prev_event)
    event_labels = {
        150: 'preceding_freekick', 154: 'preceding_corner', 157: 'preceding_save', 
        161: 'preceding_penalty', 172: 'preceding_blocked_shot', 1029: 'preceding_dangerous_attack'
    }
    df['prev_type'] = df['prev_type'].map(event_labels).fillna('preceding_other')
    df = pd.concat([df, pd.get_dummies(df['prev_type'])], axis=1)
    
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

    # attacks / min
    df.loc[(df['type'] == 1126) & (df['side'] == 'home'), 'attacks_home_cum'] = 1
    df.loc[(df['type'] == 1126) & (df['side'] == 'away'), 'attacks_away_cum'] = 1
    df[['attacks_home_cum','attacks_away_cum']] = df[['attacks_home_cum','attacks_away_cum']].fillna(0)
    df[['attacks_home_cum','attacks_away_cum']] = df.groupby('match_id')[['attacks_home_cum', 'attacks_away_cum']].cumsum().div(df['minutes'].replace(0,1), axis=0)

    # dangerous attacks / min
    df.loc[(df['type'] == 1029) & (df['side'] == 'home'), 'dangerous_attacks_home_cum'] = 1
    df.loc[(df['type'] == 1029) & (df['side'] == 'away'), 'dangerous_attacks_away_cum'] = 1
    df[['dangerous_attacks_home_cum','dangerous_attacks_away_cum']] = df[['dangerous_attacks_home_cum','dangerous_attacks_away_cum']].fillna(0)
    df[['dangerous_attacks_home_cum','dangerous_attacks_away_cum']] = df.groupby('match_id')[['dangerous_attacks_home_cum', 'dangerous_attacks_away_cum']].cumsum().div(df['minutes'].replace(0,1), axis=0)

    # turnover / min
    df['prev_side'] = df.groupby('match_id')['side'].shift()
    home_loses_possession = (df['prev_side'] == 'home') & (df['side'] == 'away')
    away_loses_possession = (df['side'] == 'home') & (df['prev_side'] == 'away')
    df.loc[home_loses_possession | away_loses_possession, 'turnover_cum'] = 1
    df['turnover_cum'] = df['turnover_cum'].fillna(0)
    df['turnover_cum'] = df.groupby('match_id')['turnover_cum'].cumsum()
    df['turnover_cum']= df['turnover_cum'].div(df['minutes'].replace(0,1), axis=0)
    df.drop(['prev_side'], axis=1, inplace=True)

    return df

def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    score = df["matchscore"].str.split(":", n = 1, expand = True)

    df["goals_home"] = (score[0]).astype(int)
    df["goals_away"] = (score[1]).astype(int)

    df["goal_diff"] = np.abs(df["goals_home"] - df["goals_away"])
    
    
    df["home_lead"] = (df["goals_home"] > df["goals_away"]).astype(int)
    df["away_lead"] = (df["goals_away"] > df["goals_home"]).astype(int)

    df["min_remaining"] = 90 - df["minutes"]
    df["goals_up_x_remaining"] = df["min_remaining"] * df["goal_diff"]

    return df.drop(['goals_home', 'goals_away'], axis=1)