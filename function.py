import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import os

def add_on_target_prob(df: pd.DataFrame, modelchoice = "linear") -> pd.DataFrame:
    on_target_ids = [30, 155]

    df["on_target"] = (df["type"].isin(on_target_ids)).astype(int)

    df["is_away"] = 1 - df.is_home

    data = df[["distance", "angle", "on_target", "dangerous_attacks_home_cum", "dangerous_attacks_away_cum", "is_home", "is_away"]]

    data_train, data_test = train_test_split(data, test_size = 0.2, random_state = 43)

    formula_string = "on_target ~ distance + angle + distance * angle + dangerous_attacks_home_cum * is_home + dangerous_attacks_away_cum * is_away"

    if modelchoice == "linear":
        model = smf.ols(formula = formula_string, data = data_train)
        result = model.fit()
    elif modelchoice == "logit":
        model = smf.logit(formula = formula_string, data = data_train)
        result = model.fit()

    print(result.summary()) 
        
    df["on_target_pred"] = result.predict(df)

    return df



