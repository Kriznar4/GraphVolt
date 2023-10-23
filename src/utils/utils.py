import pandas as pd
import os

def read_network_data(trafo_id, depth=1):
    """
    Reads all csv files from a given transformer stations. Depth is number of parent filders to get to GraphVold folder.
    """
    #get parent dir of cwd
    parent_dir = os.getcwd()
    for _ in range(depth):
        parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    #get data folder and then to networks_raw_folder
    path_data_raw = os.path.join(parent_dir, 'data', 'networks_data_raw')

    tablenames = ["bus", "bus_geodata", "EnergijaSMM", "ext_grid", "line", "line_geodata", "load", "NapetostiSMM", "tehnicni", "TPnapetosti_energija", "trafo"]

    #get path to network
    path_network = os.path.join(path_data_raw, f"{trafo_id}_anon")

    #read all csv files from path_network
    df_network_dict = {}
    for tablename in tablenames:
        path_table = os.path.join(path_network, f"{trafo_id}_{tablename}.csv")
        df_network_dict[tablename] = pd.read_csv(path_table, sep=",", decimal=".")
    
    return df_network_dict
