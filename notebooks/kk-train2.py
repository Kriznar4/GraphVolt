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
    
def train_test(model, device, train_dataset, test_dataset, optimizer, loss_fn, epochs, now):
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
                loss = loss_fn()(out, snapshot.y).cpu().numpy()
                epoch_loss_test += loss
            epoch_losses_test.append(epoch_loss_test)
            if min(epoch_losses_test) == epoch_loss_test:
                torch.save(model.state_dict(), f"../models/A3TGCN_{now}.pt")
            print("Epoch: {}, Train Loss: {:.7f}, Test Loss: {:.7f}".format(epoch, epoch_loss_train, epoch_loss_test))
        
        
    return epoch_losses_train, epoch_losses_test
            
def eval(model, eval_dataset, device, loss_fn, std):
    with torch.no_grad():
        model.eval()
        loss_all = 0
        loss_elementwise = 0
        
        steps = 0
        for snapshot in tqdm(eval_dataset, desc="Evaluating"):
            steps += 1
            snapshot.to(device)
            out = model(snapshot.x, snapshot.edge_index)
            loss_all += loss_fn()(out, snapshot.y).cpu().numpy()
            loss_elementwise += loss_fn(reduction="none")(out, snapshot.y).cpu().numpy()
        loss_all *= std/steps
        loss_elementwise *= std/steps
    return loss_all, loss_elementwise
        

torch.cuda.empty_cache() 

trafo_id = "T1330"
epochs = 1
num_timesteps_in = 12
num_timesteps_out = 4
train_ratio = 0.7
test_ratio_vs_eval_ratio = 0.5
learning_rate = 0.01

#get dateime string of now
now = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

print("Loading data...")
loader = SimpleGraphVoltDatasetLoader(trafo_id)
loader_data = loader.get_dataset(num_timesteps_in=num_timesteps_in, num_timesteps_out=num_timesteps_out)

train_dataset, test_eval_dataset = temporal_signal_split(loader_data, train_ratio=train_ratio)
test_dataset, eval_dataset = temporal_signal_split(test_eval_dataset, train_ratio=test_ratio_vs_eval_ratio)

print("Running training...")
device = torch.device('cuda')
model = TemporalGNN(node_features=train_dataset[0].x.shape[1], periods=train_dataset[0].y.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.L1Loss
losses = train_test(model, device, train_dataset, test_dataset, optimizer, loss_fn, epochs=epochs, now=now)

print(losses)

std = loader.mean_and_std["measurements"][1]["voltage"]

#read saved model
model.load_state_dict(torch.load(f"../models/A3TGCN_{now}.pt"))

loss_all, loss_elementwise = eval(model, eval_dataset, device, loss_fn, std)

print("Loss all: {:.7f}".format(loss_all))
print("Loss elementwise: {}".format(loss_elementwise))