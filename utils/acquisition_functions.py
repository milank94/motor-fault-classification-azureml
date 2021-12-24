import pandas as pd


def read_data(path_names: list) -> list:
    '''
    Reads in raw data from .csv files and returns a list
    params:
    ---
    path_names (list): list of all the data files to read in
    returns:
    ---
    sequences (list): raw dataset from data directory
    '''

    sequences = list()

    for name in path_names:
        data = pd.read_csv(name, header=None)
        sequences.append(data.values)

    return sequences
