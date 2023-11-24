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
## Functions for checking data
#------------------------------------------
def check_data(data, start_date="2021-06-01 00:00:00", end_date="2023-06-01 00:00:00", freq="15min"):
    """
    Check if the data is valid.

    There should be all records form including `start_date` to excluding 
    `end_date` with frequency `freq` for all nodes and for trafo.

    Also check if there are all smms from nodes is in timeseries.
    """

    df_nodes = data["nodes_static_data"]
    df_edges = data["edges_static_data"]
    df_smm_measurments = data["SMM_measurements"]
    df_tp_measurments = data["TP_measurements"]

    # Check if there are all smms from nodes is in timeseries.
    smms = df_nodes["smm"].unique()
    smms_in_ts = df_smm_measurments["SMM"].unique()
    missing_smms = [smm for smm in smms if smm not in smms_in_ts]
    if len(missing_smms) > 0:
        if len(missing_smms) != 1 and not missing_smms[0].isna():
            print(f"Missing smms in timeseries: {missing_smms}")

    # Check if there are all records form including `start_date` to excluding
    # `end_date` with frequency `freq` for all nodes and for trafo.
    #produce timeseries from start_date to end_date with freq
    timeseries = pd.date_range(start=start_date, end=end_date, freq=freq)[:-1]
    df_smm_grouped = df_smm_measurments.groupby("SMM")
    for i, df in df_smm_grouped:
        if timeseries.isin(df["date_time"]).sum() != len(timeseries):
            print(f"Missing records for SMM {i}")
        #check for duplicated records
        if df.duplicated(subset="date_time").sum() > 0:
            print(f"Duplicated records for SMM {i}")

    #TODO check for duplicated records
        
    if timeseries.isin(df_tp_measurments["date_time"]).sum() != len(timeseries):
        print("Missing records for trafo")
    if df_tp_measurments.duplicated(subset="date_time").sum() > 0:
            print(f"Duplicated records for SMM {i}")

    for i, df in df_smm_grouped:
        #count nan values for each column separately
        print(i)
        print(df.isna().sum())
        print("")      



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

def preprocess(data_preprocess):
    """
    Preprocesses the raw data to be ready for use as graph.
    """
    data_preprocessed_basic = basic_preprocessing(data_preprocess)
    data_preprocessed = dict()
    data_preprocessed["nodes_static_data"] = data_preprocessed_basic["nodes_static_data"]
    data_preprocessed["edges_static_data"] = data_preprocessed_basic["edges_static_data"]

    ###join tp and smm measurements
    data_ts_smm_proc = data_preprocessed_basic["SMM_measurements"]
    data_ts_tp_proc = data_preprocessed_basic["TP_measurements"]

    #TS measurements first need come extra columns
    data_ts_tp_proc = data_ts_tp_proc.rename(columns={"trafo_node_id": "node_id"})
    
    #SMM measurements need to drop some columns
    data_ts_smm_proc = data_ts_smm_proc.drop(columns=["trafo_node_id", "active_energy", "reactive_energy"])

    #we forgot te get weather data for trafo location, so we will asign it mean values of whole network for particular timestamp
    # df_mean = data_ts_smm_proc.drop(columns=["node_id", "active_power", "reactive_power", "current", "voltage", ]).groupby(["date_time"]).agg("mean").reset_index()
    # data_ts_tp_proc = pd.merge(data_ts_tp_proc, df_mean, on="date_time", how="inner")

    #join tp and smm measurements to one dataframe
    df_ts_smm_tp_proc = pd.concat([data_ts_smm_proc, data_ts_tp_proc], ignore_index=True)

    #since almost all data about current is missing we will drop it
    df_ts_smm_tp_proc = df_ts_smm_tp_proc.drop(columns=["current"])

    #TODO fill nans

    #we now have to aggregate data by date_time and node_id as one node can have multiple measurements at the same time (multiple SMM for one PMO)
    #some columns will be aggregated by mean, some by sum
    df_mean = df_ts_smm_tp_proc.drop(columns=["active_power", "reactive_power"])
    df_sum = df_ts_smm_tp_proc[["date_time", "node_id", "active_power", "reactive_power"]]

    df_mean = df_mean.groupby(["date_time", "node_id"]).agg("mean").reset_index()
    df_sum = df_sum.groupby(["date_time", "node_id"]).agg("sum").reset_index()

    df_ts_smm_tp_proc = pd.merge(df_mean, df_sum, on=["date_time", "node_id"], how="inner")

    #add to data dictionary
    data_preprocessed["measurements"] = df_ts_smm_tp_proc

    return data_preprocessed