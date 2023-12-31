import pandas as pd
import numpy as np
import sys
sys.path.append("../src/utils")
from utils import SimpleGraphVoltDatasetLoader_Lazy
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        out_channels = 32
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



def train_test(model,device, train_dataset, test_dataset, optimizer, loss_fn, epochs, now):
    """
    Definition of the training loop.
    """
    epoch_losses_train = []
    epoch_losses_test = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss_train = 0

        for snapshot_i in tqdm(train_dataset, desc="Training epoch {}".format(epoch)):
            snapshot = loader.get_snapshot(snapshot_i)
            snapshot.to(device)
            optimizer.zero_grad()
            out = model(snapshot.x, snapshot.edge_index,snapshot.edge_attr)
            loss = loss_fn()(out, snapshot.y)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.detach().cpu().numpy()

        epoch_losses_train.append(epoch_loss_train)

        model.eval()
        epoch_loss_test = 0
        with torch.no_grad():

            for snapshot_j in tqdm(test_dataset, desc="Testing epoch {}".format(epoch)):
                snapshot = loader.get_snapshot(snapshot_j)
                snapshot.to(device)
                out = model(snapshot.x, snapshot.edge_index,snapshot.edge_attr)
                loss = loss_fn()(out, snapshot.y).cpu().numpy()
                epoch_loss_test += loss

            epoch_losses_test.append(epoch_loss_test)
            if min(epoch_losses_test) == epoch_loss_test:
                torch.save(model.state_dict(), f"../models/A3TGCN_{now}_{trafo_id}_epochs-{epochs}_in-{num_timesteps_in}_out-{num_timesteps_out}_train-ratio-{train_ratio}_lr-{learning_rate}.pt")
            print("Epoch: {}, Train Loss: {:.7f}, Test Loss: {:.7f}".format(epoch, epoch_loss_train, epoch_loss_test))
        
        
    return epoch_losses_train, epoch_losses_test
            
            
            
            
def eval(model, loader, eval_dataset, device, loss_fn, std):
    with torch.no_grad():
        model.eval()
        loss_all = 0
        loss_elementwise = 0
        
        steps = 0
        for snapshot_i in tqdm(eval_dataset, desc="Evaluating"):
            snapshot = loader.get_snapshot(snapshot_i)
            steps += 1
            snapshot.to(device)
            out = model(snapshot.x, snapshot.edge_index,snapshot.edge_attr)
            loss_all += loss_fn()(out, snapshot.y).cpu().numpy()
            loss_elementwise += loss_fn(reduction="none")(out, snapshot.y).cpu().numpy()
        loss_all *= std/steps
        loss_elementwise *= std/steps
    return loss_all, loss_elementwise
#------parameters------ 

trafo_id = "T1330"
epochs = 25
num_timesteps_in = 12
num_timesteps_out = 4
train_ratio = 0.7
test_ratio_vs_eval_ratio = 0.5
learning_rate = 0.01
device_str = 'cpu'

#----------------------
if device_str == 'cpu':
    torch.cuda.empty_cache()

#get dateime string of now
now = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

name = f"../models/A3TGCN_{now}_{trafo_id}_epochs-{epochs}_in-{num_timesteps_in}_out-{num_timesteps_out}_train-ratio-{train_ratio}_lr-{learning_rate}.pt"

print("Loading data...")
loader = SimpleGraphVoltDatasetLoader_Lazy(trafo_id, num_timesteps_in, num_timesteps_out)
print(" done")
loader_data_index = loader.snapshot_index

train_dataset, test_eval_dataset = loader.temporal_signal_split_lazy_cut(loader_data_index)
test_dataset, eval_dataset = loader.temporal_signal_split_lazy(test_eval_dataset, train_ratio=test_ratio_vs_eval_ratio)

print("Running training...")
device = torch.device(device_str)
model = TemporalGNN(node_features=loader.num_features, edge_features=loader.num_edge_features ,periods=num_timesteps_out).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss
losses = train_test(model, device, train_dataset, test_dataset, optimizer, loss_fn, epochs=epochs, now=now)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

print(losses)

std = loader.mean_and_std["measurements"][1]["voltage"]

#read saved model
model.load_state_dict(torch.load(name))

loss_all, loss_elementwise = eval(model, loader, eval_dataset, device, loss_fn, std)

print("Loss all: {:.7f}".format(loss_all))
print("Loss elementwise: {}".format(loss_elementwise))