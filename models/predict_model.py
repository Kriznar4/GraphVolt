import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from utils import read_raw_network_data
from utils import preprocess

class GraphVoltDatasetLoader(object):
    '''
        Check this https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/wikimath.html#WikiMathsDatasetLoader
        for an example of how to implement a dataset loader

        And here are the docs https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/signal.html
    '''
    def __init__(self,trafo):
        data = preprocess(read_raw_network_data(trafo,depth=1))
        self.nodes_data = data['nodes_static_data']
        self.edges_data = data['measurements']
        self.ts_data = data['ts_data']
    
    def _get_edges(self):
        self._edges = self.edges_data[["from_node_id", "to_node_id"]].values.T
    
    def _get_edges_weights(self):
        self._edges_weights = self.edges_data.drop(["from_node_id", "to_node_id"], axis=1).values

    def _get_features(self):
        ### TODO ###
        pass

    def _get_targets(self):
        ### TODO ###
        pass

    def _get_dataset(self):
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset