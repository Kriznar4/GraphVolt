import pandas as pd
import numpy as np
import os

def read_raw_network_data(trafo_id, depth=1):
    """
    Reads all csv files from a given transformer stations. Depth is number of parent filders to get to GraphVolt folder.
    """
    #get parent dir of cwd
    parent_dir = os.getcwd()
    for _ in range(depth):
        parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    #get data folder and then to networks_raw_folder
    path_data_raw = os.path.join(parent_dir, 'data', 'networks_data_raw')

    tablenames = ["edges_static_data", "nodes_static_data", "SMM_measurements", "TP_measurements"]

    #get path to network
    path_network = os.path.join(path_data_raw, f"{trafo_id}_anon_procesed")

    #read all csv files from path_network
    df_network_dict = {}
    for tablename in tablenames:
        path_table = os.path.join(path_network, f"{trafo_id}_{tablename}.csv")
        df_network_dict[tablename] = pd.read_csv(path_table, sep=",", decimal=".")
    
    return df_network_dict, path_network

def save_raw_network_data(trafo_id, data, depth=1):
    """
    Reads all csv files from a given transformer stations. Depth is number of parent filders to get to GraphVolt folder.
    """
    #get parent dir of cwd
    parent_dir = os.getcwd()
    for _ in range(depth):
        parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    #get data folder and then to networks_raw_folder
    path_data_raw = os.path.join(parent_dir, 'data', 'networks_data_raw')

    tablenames = ["edges_static_data", "nodes_static_data", "SMM_measurements", "TP_measurements"]

    #get path to network
    path_network = os.path.join(path_data_raw, f"{trafo_id}_anon_procesed")

    #read all csv files from path_network
    df_network_dict = {}
    for tablename in tablenames:
        path_table = os.path.join(path_network, f"{trafo_id}_{tablename}.csv")
        # df_network_dict[tablename] = pd.read_csv(path_table, sep=",", decimal=".")
        data[tablename].to_csv(path_table, sep=",", decimal=".", index=False)
        
    
    return trafo_id

#------------------------------------------
## Functions for basic data preprocessing
#------------------------------------------
### Helper functions
#------------------------------------------
#### Node preprocessing
def preprocess_nodes(data_nodes_proc):
    """
    Preprocesses nodes data and returns a dataframe with the aggregated data and dictionaries for mapping from bus and smm_list to nodeid.
    """
    
    #copy of code from 02-kk-create-nx.ipynb
    data_nodes_aggr = data_nodes_proc.groupby(["bus", "aclass_id"]).agg(list)
    data_nodes_aggr = data_nodes_aggr.rename(columns={"smm": "smm_list"})
    data_nodes_aggr["x_y"] = data_nodes_aggr["x_y"].apply(lambda x: x[0])
    data_nodes_aggr["lon_lat"] = data_nodes_aggr["lon_lat"].apply(lambda x: x[0])
    data_nodes_aggr = data_nodes_aggr.reset_index()
    
    #dictionaries for mapping from bus and smm_list to nodeid
    dict_of_bus_to_nodeid = dict(zip(data_nodes_aggr["bus"].tolist(), range(len(data_nodes_aggr))))
    dict_of_smms_to_nodeid =  { x: i for i, y in enumerate(data_nodes_aggr["smm_list"].tolist()) if (y!= None) for x in y }
    
    #map bus to node_id
    data_nodes_aggr.insert(0, "node_id", range(len(data_nodes_aggr)))
    data_nodes_aggr["node_id"] = data_nodes_aggr["bus"].apply(lambda x: dict_of_bus_to_nodeid[x])
    
    #aggregate prikljucna_moc_odjem to prikljucna_moc_odjem_aggr
    data_nodes_aggr.insert(5, "prikljucna_moc_odjem_aggr", data_nodes_aggr["prikljucna_moc_odjem"].apply(sum))
    
    #aggregate prikljucna_moc_oddaja to prikljucna_moc_oddaja_aggr
    data_nodes_aggr.insert(7, "prikljucna_moc_oddaja_aggr", data_nodes_aggr["prikljucna_moc_oddaja"].apply(sum))
    
    data_nodes_aggr = data_nodes_aggr.drop(columns=["bus"])
    
    return data_nodes_aggr, dict_of_bus_to_nodeid, dict_of_smms_to_nodeid

#### Edge preprocessing
def preprocess_edges(data_edges_proc, dict_bus2nodeid):
    """
    Preprocesses edge data. Replaces from_bus and to_bus with from_node_id and to_node_id.
    """
    
    #insert from_node_id and set them to corresponding node_id from bus. Drop from_bus.
    data_edges_proc.insert(0, "from_node_id", range(len(data_edges_proc)))
    data_edges_proc["from_node_id"] = data_edges_proc["from_bus"].apply(lambda x: dict_bus2nodeid[x])
    data_edges_proc = data_edges_proc.drop(columns=["from_bus"])
    
    #insert to_node_id and set them to corresponding node_id from bus. Drop to_bus.
    data_edges_proc.insert(1, "to_node_id", range(len(data_edges_proc)))
    data_edges_proc["to_node_id"] = data_edges_proc["to_bus"].apply(lambda x: dict_bus2nodeid[x])
    data_edges_proc = data_edges_proc.drop(columns=["to_bus"])
    
    return data_edges_proc

#### TP timeseries preprocessing
def preprocess_ts_tp(data_ts_tp_proc, dict_bus2nodeid):
    """
    Preprocesses time series data for TP measurements. Replaces trafo_id and trafo_station_id with trafo_node_id.
    """
    
    #insert trafo_node_id and set them to corresponding node_id from bus. Drop trafo_id and trafo_station_id.
    data_ts_tp_proc.insert(0, "trafo_node_id", np.zeros(len(data_ts_tp_proc)))
    data_ts_tp_proc.drop(columns=["trafo_id", "trafo_station_id"], inplace=True)
    
    return data_ts_tp_proc

#### SMM timeseries preprocessing
def preprocess_ts_smm(data_ts_smm_proc, dict_smms2nodeid, dict_bus2nodeid):
    """
    Preprocesses time series data for SMM measurements. Replaces SMM with node_id and trafo_station_id with trafo_node_id.
    """
    
    #insert node_id and set them to corresponding node_id from bus. Drop SMM.
    data_ts_smm_proc.insert(0, "node_id", range(len(data_ts_smm_proc)))
    data_ts_smm_proc["node_id"] = data_ts_smm_proc["SMM"].apply(lambda x: dict_smms2nodeid[x])
    data_ts_smm_proc = data_ts_smm_proc.drop(columns=["SMM"])
    
    #if trafo_station_id corresponds to bus, map it to node_id.
    # data_ts_smm_proc.insert(0, "trafo_station_node_id", range(len(data_ts_smm_proc)))
    # data_ts_smm_proc["trafo_station_node_id"] = data_ts_smm_proc["trafo_station_id"].map(lambda x: dict_bus2nodeid[x])
    
    #insert trafo_node_id and set them to corresponding node_id from bus. Drop trafo_id and trafo_station_id.
    data_ts_smm_proc.insert(1, "trafo_node_id", np.zeros(len(data_ts_smm_proc)))
    data_ts_smm_proc.drop(columns=["trafo_id", "trafo_station_id"], inplace=True)
    
    return data_ts_smm_proc

#------------------------------------------
### Main function
def basic_preprocessing(data_preprocess):
    """
    Preprocesses the data.
    """
    data_preprocessed = data_preprocess.copy()
    data_nodes_proc = data_preprocessed["nodes_static_data"]
    data_edges_proc = data_preprocessed["edges_static_data"]
    data_ts_smm_proc = data_preprocessed["SMM_measurements"]
    data_ts_tp_proc = data_preprocessed["TP_measurements"]
    
    data_nodes_proc, dict_bus2nodeid, dict_smms2nodeid = preprocess_nodes(data_nodes_proc)
    trafo_node_id = 0
    data_edges_proc = preprocess_edges(data_edges_proc, dict_bus2nodeid)
    data_ts_smm_proc = preprocess_ts_smm(data_ts_smm_proc, dict_smms2nodeid, dict_bus2nodeid)
    data_ts_tp_proc = preprocess_ts_tp(data_ts_tp_proc, dict_bus2nodeid)
    
    data_preprocessed["nodes_static_data"] = data_nodes_proc
    data_preprocessed["edges_static_data"] = data_edges_proc
    data_preprocessed["SMM_measurements"] = data_ts_smm_proc
    data_preprocessed["TP_measurements"] = data_ts_tp_proc
    
    return data_preprocessed  #, dict_bus2nodeid, dict_smms2nodeid
