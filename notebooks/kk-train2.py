import pandas as pd
import numpy as np
import sys
sys.path.append("../src/utils")
from utils import SimpleGraphVoltDatasetLoader, read_and_prepare_data
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
from tqdm import tqdm


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        out_channels = 32
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
    
def train_test(model, device, train_dataset, test_dataset, optimizer, loss_fn, epochs):
    """
    Definition of the training loop.
    """
    epoch_losses_train = []
    epoch_losses_test = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss_train = 0
        for snapshot in tqdm(train_dataset, desc="Training epoch {}".format(epoch)):
            snapshot.to(device)
            optimizer.zero_grad()
            out = model(snapshot.x, snapshot.edge_index)
            loss = loss_fn()(out, snapshot.y)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.detach().cpu().numpy()
        epoch_losses_train.append(epoch_loss_train)
        model.eval()
        epoch_loss_test = 0
        with torch.no_grad():
            for snapshot in tqdm(test_dataset, desc="Testing epoch {}".format(epoch)):
                snapshot.to(device)
                out = model(snapshot.x, snapshot.edge_index)
                loss = loss_fn()(out, snapshot.y)
                epoch_loss_test += loss.detach().cpu().numpy()
            epoch_losses_test.append(epoch_loss_test)
            print("Epoch: {}, Train Loss: {:.7f}, Test Loss: {:.7f}".format(epoch, epoch_loss_train, epoch_loss_test))
        
        
    return epoch_losses_train, epoch_losses_test
            
    
        

torch.cuda.empty_cache() 

trafo_id = "T1330"

print("Loading data...")
loader = SimpleGraphVoltDatasetLoader(trafo_id)
loader_data = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=4)

train_dataset, test_eval_dataset = temporal_signal_split(loader_data, train_ratio=0.7)
test_dataset, eval_dataset = temporal_signal_split(test_eval_dataset, train_ratio=0.5)

print("Running training...")
device = torch.device('cuda')
model = TemporalGNN(node_features=train_dataset[0].x.shape[1], periods=train_dataset[0].y.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss
losses = train_test(model, device, train_dataset, test_dataset, optimizer, loss_fn, epochs=2)

print(losses)