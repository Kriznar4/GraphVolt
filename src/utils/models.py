import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from pytorch_geometri_temporal.nn.recurrent import A3TGCN, GConvLSTM

class avgModel():
    def __init__(self, num_timesteps_out, voltage_index):
        self.num_timesteps_out = num_timesteps_out
        self.voltage_index = voltage_index

    def __call__(self, x, edge_index):
        predicted_voltages = torch.mean(x[:, self.voltage_index, :], dim=1, keepdim=True)
        return predicted_voltages.repeat(1, self.num_timesteps_out)

    def train(self):
        pass

    def eval(self):
        pass



class RNN_LSTM(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch, num_layers):
        super(RNN_LSTM, self).__init__()
        self.lstm = nn.LSTM(in_ch, hidden_ch, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_ch, out_ch)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        # x = self.fc(x[-1, :])
        return x
    


class GNN_A3TGCN(torch.nn.Module):
    def __init__(self, node_features, periods, hidden=32):
        super(GNN_A3TGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        out_channels = hidden
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=out_channels, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h
    


class GNN_A3TGCN_ea(torch.nn.Module):
    def __init__(self, node_features, periods, hidden=32):
        super(GNN_A3TGCN_ea, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        out_channels = hidden
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=out_channels, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index,edge_attr)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
class GNN_GCNLSTM(torch.nn.Module):
    def __init__(self,node_features, periods, hidden=32):
        super(GNN_GCNLSTM, self).__init__()

        out_channels= hidden
        K = 5 # size of Chebyshev filter
        self.recurrent_1 = GConvLSTM(
            in_channels=node_features, 
            out_channels=out_channels, 
            K=K, normalization='sym', 
            bias=False)
        
        # self.recurrent_2 = GConvLSTM(
        #     in_channels=out_channels, 
        #     out_channels=out_channels, 
        #     K=K, normalization='sym', 
        #     bias=False)

        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, timesteps, edge_index):

        h1, c1 = None, None
        for x in timesteps:
            h1, c1 = self.recurrent_1(x, edge_index, H=h1, C=c1)
            #h2, c2 = self.recurrent_2(h1, edge_index, H=h2, C=c2)

        x = F.relu(h1)
        x = self.linear(x)

        return x
    


    
class GNN_GCNLSTM_ea_fs(torch.nn.Module):
    def __init__(self, node_features, edge_features, periods, hidden):
        super(GNN_GCNLSTM_ea_fs, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        out_channels = hidden
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=out_channels, 
                           periods=periods)
        
        self.feature_linear = torch.nn.Linear(node_features, node_features)

        self.edge_features_linear = torch.nn.Linear(edge_features, 1)

        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels,periods)

    def forward(self, x, edge_index, edge_features):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        edge_weights = Graph edge weights
        """

        x = x.permute(2, 0, 1)
        x = self.feature_linear(x)
        x = x.permute(1, 2, 0)

        edge_features = self.edge_features_linear(edge_features).squeeze()
        edge_features = F.relu(edge_features)

        h = self.tgnn(x, edge_index,edge_weight=edge_features)

        h = F.relu(h)
        h = self.linear(h)
        return h