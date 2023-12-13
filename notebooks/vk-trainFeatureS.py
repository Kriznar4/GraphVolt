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

class FeatureMLP(torch.nn.Module):
    def __init__(self, input_size,num_timesteps_in):
        super(FeatureMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size*num_timesteps_in, num_timesteps_in),
            torch.nn.Sigmoid() 
        )

    def forward(self, x):
        return self.mlp(x)

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, num_timesteps_in):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        out_channels = 32
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=out_channels, 
                           periods=periods)
        
        self.feature_mlp = FeatureMLP(node_features,num_timesteps_in)

        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels,periods)

    def forward(self, x, edge_index, edge_weights):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        edge_weights = Graph edge weights
        """

        weighted_x = torch.zeros(x.size(0),x.size(1),x.size(2))

        for i in range(x.size(0)):
                
            node = x[i,:,:]
            reshaped_node = node.reshape(1,-1)
            feature_weights = self.feature_mlp(reshaped_node)
            weighted_x[i] = node * feature_weights

        #we can do this for edge_weights too...but then we need another MLP

        h = self.tgnn(weighted_x, edge_index, edge_weights)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
def train_test(model, feature_mlp,device, train_dataset, test_dataset, optimizer, loss_fn, epochs, now):
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

            #create a tensor of the same size as the snapshot.x where I will store the weighted x
            weighted_x = torch.zeros(snapshot.x.size(0),snapshot.x.size(1),snapshot.x.size(2))

            for i in range(snapshot.x.size(0)):
                
                node = snapshot.x[i,:,:]
                reshaped_node = node.reshape(1,-1)
                feature_weights = feature_mlp(reshaped_node)
                weighted_x[i] = node * feature_weights

            optimizer.zero_grad()
            out = model(weighted_x, snapshot.edge_index,snapshot.edge_weight)
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

                weighted_x = torch.zeros(snapshot.x.size(0),snapshot.x.size(1),snapshot.x.size(2))
                for i in range(snapshot.x.size(0)): 
                    node = snapshot.x[i,:,:]
                    reshaped_node = node.reshape(1,-1)
                    feature_weights = feature_mlp(reshaped_node)

                    weighted_x[i] = node * feature_weights

                out = model(weighted_x, snapshot.edge_index,snapshot.edge_weight)
                loss = loss_fn()(out, snapshot.y).cpu().numpy()
                epoch_loss_test += loss

            epoch_losses_test.append(epoch_loss_test)
            if min(epoch_losses_test) == epoch_loss_test:
                torch.save(model.state_dict(), f"../models/A3TGCN_{now}_{trafo_id}_epochs-{epochs}_in-{num_timesteps_in}_out-{num_timesteps_out}_train-ratio-{train_ratio}_lr-{learning_rate}.pt")
            print("Epoch: {}, Train Loss: {:.7f}, Test Loss: {:.7f}".format(epoch, epoch_loss_train, epoch_loss_test))
        
        
    return epoch_losses_train, epoch_losses_test
            
            
def eval(model, feature_mlp,eval_dataset, device, loss_fn, std):
    with torch.no_grad():
        model.eval()
        loss_all = 0
        loss_elementwise = 0
        
        for snapshot in tqdm(eval_dataset, desc="Evaluating"):
            snapshot.to(device)

            weighted_x = torch.zeros(snapshot.x.size(0),snapshot.x.size(1),snapshot.x.size(2))
            for i in range(snapshot.x.size(0)): 
                node = snapshot.x[i,:,:]
                reshaped_node = node.reshape(1,-1)
                feature_weights = feature_mlp(reshaped_node)

                weighted_x[i] = node * feature_weights

            out = model(weighted_x, snapshot.edge_index,snapshot.edge_weight)
            loss_all += loss_fn()(out, snapshot.y).cpu().numpy()
            loss_elementwise += loss_fn(reduction="none")(out, snapshot.y).cpu().numpy()

        loss_all *= std/steps
        loss_elementwise *= std/steps
    return loss_all, loss_elementwise
#------parameters------ 

trafo_id = "T1330"
epochs = 30
num_timesteps_in = 12
num_timesteps_out = 4
train_ratio = 0.7
test_ratio_vs_eval_ratio = 0.5
learning_rate = 0.01
device_str = 'cpu'
hidden = 32

#----------------------
if device_str == 'cuda':
    torch.cuda.empty_cache()

#get dateime string of now
now = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

name = f"../models/A3TGCN_{now}_{trafo_id}_epochs-{epochs}_in-{num_timesteps_in}_out-{num_timesteps_out}_train-ratio-{train_ratio}_lr-{learning_rate}_hidden-{hidden}.pt"

print("Loading data...")
loader = SimpleGraphVoltDatasetLoader_Lazy(trafo_id, num_timesteps_in, num_timesteps_out)
print(" done")
loader_data_index = loader.snapshot_index

train_dataset, test_eval_dataset = loader.temporal_signal_split_lazy(loader_data_index, train_ratio=train_ratio)
test_dataset, eval_dataset = loader.temporal_signal_split_lazy(test_eval_dataset, train_ratio=test_ratio_vs_eval_ratio)

print("Running training...")
device = torch.device(device_str)
feature_mlp = FeatureMLP(train_dataset[0].x.shape[1],num_timesteps_in=num_timesteps_in).to(device)
model = TemporalGNN(node_features=train_dataset[0].x.shape[1], periods=train_dataset[0].y.shape[1],num_timesteps_in=num_timesteps_in).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.L1Loss
losses = train_test(model, feature_mlp, device, train_dataset, test_dataset, optimizer, loss_fn, epochs=epochs, now=now)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

print(losses)

std = loader.mean_and_std["measurements"][1]["voltage"]

#read saved model
model.load_state_dict(torch.load(name))

loss_all, loss_elementwise = eval(model, loader, eval_dataset, device, loss_fn, std)

print("Loss all: {:.7f}".format(loss_all))
print("Loss elementwise: {}".format(loss_elementwise))