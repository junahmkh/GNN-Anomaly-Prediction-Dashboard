import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_file(node_path):
    node_data = pd.read_parquet(node_path)
    node_data = node_data.dropna()
    return node_data

def get_node_name(file_path):
    tmp = file_path.split("/")
    tmp = tmp[-1].split(".")
    node = tmp[0]

    return int(node)

def data_fetch(rack, ts):
    data_dir = f"/data/{rack}/"

    with open("col_list.pickle","rb") as f:
        cols = pickle.load(f)

    files = []
    # loop over the contents of the directory
    for filename in os.listdir(data_dir):
        # construct the full path of the file
        file_path = os.path.join(data_dir, filename)
        # check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            files.append(file_path)

    # Sort using the custom function
    sorted_files = sorted(files, key=get_node_name)
    
    fetched_data_df = pd.DataFrame(columns=cols)
    for file_path in sorted_files:
        df = read_file(file_path)
        df_ts = df[df['timestamp'] == ts]
        fetched_data_df = pd.concat([fetched_data_df,df_ts])
    
    fetched_data_df = fetched_data_df.reset_index(drop=True)
    fetched_data_df = fetched_data_df.drop(columns=["timestamp"])
    return fetched_data_df.fillna(0)




