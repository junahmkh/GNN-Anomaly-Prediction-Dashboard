import pandas as pd
import numpy as np
from sklearn import preprocessing

def scale_df(df):
    scaler = preprocessing.MinMaxScaler()
    names = df.columns
    d = scaler.fit_transform(df)
    df = pd.DataFrame(d, columns=names)

    return df


def preprocess()