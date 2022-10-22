import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import os

def add_on_target_prob(df: pd.DataFrame) -> pd.DataFrame:
    on_target_ids = [30, 155]

    df = df.copy()

    df["on_target"] = (df["type"].isin(on_target_ids)).astype(int)

    data = df[["distance", "angle", "solid_angle", "on_target"]]
    data_train, data_test = train_test_split(data, test_size = 0.2, random_state = 43)

    model = smf.ols(formula = "on_target ~ distance + solid_angle + distance * solid_angle", data = data_train)
    result = model.fit()

    df["on_target_pred"] = result.predict(df)

    return df



