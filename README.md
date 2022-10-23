# Modeling expected goals in football matches

Repository for Sportradar Case 2 (ML) in Start Code Hackaton.

## Requirements

To download the required packages, run `pip install -r /path/to/requirements.txt`. 

## Pipeline

We have divided the featuers into two groups: 
- *Event features* (found in `event_features.py`): the features gotten directly from the scoring opurtunity. These include angle and distance relative to goal, as well as type of shot (header, shot, penalty). 
- *Game state features* (found in `game_state_features.py`): features describing the current state of game. These include cummulative statistics for game events, as well as the event leading up to event. 

The functions for extracting these features are pipelined in `pipeline.py`. The function transforming the input data is `transform_events(·)`.

## Inspecting the model

The model is fitted in `model_fitting.ipynb`. The pipeline corresponds to the best model found thus far, using all features. This is the model used in the dashboard (`App.py`).

Run it by navigating to the project repository and run `python App.py`.
