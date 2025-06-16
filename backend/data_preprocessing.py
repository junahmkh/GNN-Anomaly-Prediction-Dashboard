import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing

def scale_df(df):
    scaler = preprocessing.MinMaxScaler()
    names = df.columns
    d = scaler.fit_transform(df)
    df = pd.DataFrame(d, columns=names)

    return df

def make_edge_index(num_nodes):
    edges = []
    for i in range(num_nodes):
        if i > 0:
            edges.append([i, i - 1])
        if i < num_nodes - 1:
            edges.append([i, i + 1])
    # Transpose to COO format (2 x num_edges)
    edge_index = list(map(list, zip(*edges)))
    return edge_index

def pre_process(df):
    scaled_df = scale_df(df)
    df_feat = scaled_df.to_numpy()

    edge_index = make_edge_index(df.shape[0])

    return {
        "x": df_feat.tolist(),
        "edge_index": edge_index
    }

    
