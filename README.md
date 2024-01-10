# GraphVolt
Project for MLG (CS244W). Predicting voltages on a low voltage network with GNNs.

## Configurate conda
In terminal execute next commands:
```powershell
conda create -n "graphVolt" python=3.9
conda activate graphVolt
```
If last command doesn't try installing pytorch manually. And run it again.
Next time run command to activate created environment:
```powershell
conda activate graphVolt
```
What to install?

1. pip install pytorch **!!!GET VERSION 2.1.0!!!**
2. pip install pytorch geometric (and all its dependancies)
3. pip install pytorch geometric temporal 

for example 

```powershell
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install torch_geometric

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install torch-geometric-temporal
```

Link to medium article (just a preview): https://medium.com/@tt6523/short-term-voltage-forecasting-with-graph-neural-networks-ce8c81eb6e90.
