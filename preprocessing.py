import pandas as pd
import numpy as np

from scipy import integrate

GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
PITCH_WIDTH = 68
PITCH_LENGTH = 105


def get_goal_type(extrainfo: pd.Series):
    # Defaults to shot
    types = {1: 'penalty', 3: 'header'}
    return extrainfo.apply(lambda x: types.get(x, 'shot'))

# Encode shot types
def encode_shot_types(df: pd.DataFrame):

    # Goals
    df.loc[df['type'] == 30, 'goal'] = 1
    df.loc[df['type'] != 30, 'goal'] = 0

    # Goal type
    df.loc[df['type'] == 30, 'shot_type'] = get_goal_type(df['extrainfo'])
    # Shot on target
    df.loc[df['type'].isin([155, 156, 172]), 'shot_type'] = 'shot'
    df.loc[df['type'] == 666, 'shot_type'] = 'penalty'

    return df

def get_relative_coordinates(df):

    df = df.copy()

    df['posx'] = df['posx'] * PITCH_LENGTH / 100
    df['posy'] = df['posy'] * PITCH_WIDTH / 100

    home = df['side'] == 'home'

    df.loc[home, 'posx'] = PITCH_LENGTH - df.loc[home, 'posx']
    df.loc[~home, 'posx'] = df.loc[~home, 'posx']

    df.loc[home, 'posy'] = PITCH_WIDTH / 2 - df.loc[home, 'posy']
    df.loc[~home, 'posy'] = PITCH_WIDTH / 2 - df.loc[~home, 'posy']

    return df['posx'], df['posy']

def compute_if_shot_is_in_boxes(df):
    df = df.copy()
    rel_x, rel_y = get_relative_coordinates(df)
    df['is_in_penalty_area'] = ((rel_x <= 16.5) & (rel_y >= -20.16) & (rel_y <= 20.16)).astype(int)
    df['is_in_goal_area'] = ((rel_x <= 5.5) & (rel_y >= -9.16) & (rel_y <= 9.16)).astype(int)
    return df
        
def compute_distance_to_goal(df):
    """
    Computes the euclidean distance to the goal given the relative and scaled position on the pitch.  
    """
    x, y = get_relative_coordinates(df)
    distance = np.sqrt((x**2) + (y**2))
    
    return distance

def compute_angle_to_goal(df, between_goal_posts=False):
    """
    Computes the radian angle to the goal from a position on the pitch.
    """
    x, y = get_relative_coordinates(df)
    if between_goal_posts:
        dot_product = x*x + ((GOAL_WIDTH/2)+y)*(-(GOAL_WIDTH/2)+y)
        prod_magnitude = np.sqrt((x**2) + ((y+(GOAL_WIDTH/2)))**2)*np.sqrt((x**2) + ((y-(GOAL_WIDTH/2)))**2)
        theta = np.arccos(dot_product/prod_magnitude)
        return theta
    
    return np.arctan(y/x)

##--------------------------------------##

def get_scaled_coordinates(x, y):
    """
    Computes the scaled coordinates from the native percentage coordinates.
    """
    scaled_x = x*PITCH_LENGTH/100
    scaled_y = y*PITCH_WIDTH/100
    return scaled_x, scaled_y

def get_relative_coordinates_to_goal(x, y, is_home_team = True):
    """
    Calculates the relative coordinates of a scaled position to the goal given if its by the
    home team or the away team.
    """
    if is_home_team:
        rel_x = PITCH_LENGTH - x
    else:
        rel_x = x
    rel_y = (PITCH_WIDTH/2) - y
    return rel_x, rel_y
        

    
def compute_solid_angle_to_goal(rel_x, rel_y, rel_z=0.0001, is_header=False):
    """ 
    Computes the solid angle of a point on the field given the relative distances to the goal. This is used
    to quantify the scoring space given by the area within the goalmouth which is feasible for scoring. The 
    dimensionless unit of the solid angle steradian.
    """
    rel_x = max(rel_x, rel_z)
    rel_y = max(rel_y, rel_z)

    def theta_1(phi): return np.arctan(1/((rel_z+(GOAL_WIDTH/2))*np.cos(phi)/rel_x))
    def theta_2(phi): return np.arctan(1/((rel_z*np.cos(phi)/rel_x))) 
    def compute_length_weight(distance):
        a = 1
        if not is_header:
            b = 50
        else:
            b = 16.5
        return b - a*distance

    phi_1 = np.arctan((rel_y-(GOAL_WIDTH/2))/rel_x) # angle to the goal post with the smaller x coordinate
    phi_2 = np.arctan((rel_y+(GOAL_WIDTH/2))/rel_x) # angle to the goal post with the larger x coordinate
    distance = distance = np.sqrt((rel_x**2) + (rel_y**2))
    length_weight = compute_length_weight(distance) 
    f = lambda theta, phi: np.sin(theta)*length_weight
    omega = integrate.dblquad(f, phi_1, phi_2, theta_1, theta_2)
    return omega[0]


def compute_positional_features(x, y, is_home, is_header):
    scaled_x, scaled_y = get_scaled_coordinates(x, y)
    relative_x, relative_y = get_relative_coordinates_to_goal(scaled_x, scaled_y, is_home)
    solid_angle = compute_solid_angle_to_goal(relative_x, relative_y, is_header=is_header)
    return solid_angle


                        
                            
