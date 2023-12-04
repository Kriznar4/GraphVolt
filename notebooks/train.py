import pandas as pd
import numpy as np
import sys
sys.path.append("../src/utils")
from utils import SimpleGraphVoltDatasetLoader, read_and_prepare_data
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm

import torch
torch.cuda.empty_cache() 



trafo_id = "T1330"

loader = SimpleGraphVoltDatasetLoader(trafo_id)
loader_data = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=4)

train_dataset, test_dataset = temporal_signal_split(loader_data, train_ratio=0.8)



import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

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

TemporalGNN(node_features=4, periods=4) #12



# GPU support
device = torch.device('cpu') # cuda
subset = 2000

# Create model and optimizers
model = TemporalGNN(node_features=21, periods=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
batch_size = 5000

epochs_loss = []

print("Running training...")
for epoch in range(2): 
    epoch_loss = []
    loss = 0
    step = 0
    print(f"--------- Epoch {epoch} ---------")
    for snapshot in tqdm(train_dataset):
        #print(f"--- Step {step} ---")
        snapshot = snapshot.to(device)
        # Get model predictions
        y_hat = model(snapshot.x, snapshot.edge_index)
        # Mean squared error
        intermedaiate_loss = torch.mean((y_hat-snapshot.y)**2) 
        loss = loss + intermedaiate_loss
        step += 1
        # if step%1000==0:
        #   print(f"Intermediate loss at step {step}: {intermedaiate_loss.item()}")
        # # if step > subset:
        #   break
        if step%batch_size == 0:
            loss = loss / batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("Epoch {}, batch to {} train MSE: {:.4f}".format(epoch, step, loss.item()))
            epoch_loss.append(loss.detach().cpu().numpy())
            loss = 0
    else: 
        loss = loss / step%batch_size
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch {}, batch to {} train MSE: {:.4f}".format(epoch, step, loss.item()))
        epoch_loss.append(loss.detach().cpu().numpy())
        loss = 0

    # loss = loss / (step + 1)
    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    # print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
    # epochs_loss.append(loss.detach().cpu().numpy())
    
    epochs_loss.append(np.mean(epoch_loss))
    
    
    
#move rpochs_loss to cpu
#epochs_loss = [x.detach().numpy()  for x in epochs_loss]
#plot epochs_loss
# import matplotlib.pyplot as plt
# print(epochs_loss)
# plt.plot(epochs_loss)
# plt.show()
