import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import sys
sys.path.append("../src/utils")
from utils import read_raw_network_data,fill_data_simple_homogeneous,preprocess

#TODO: update it
def LossCurve(epochs, train_losses, test_losses, eval_losses):
    """
    Plot the loss curve.
    """
    plt.plot(range(epochs), train_losses, label="Train loss")
    plt.plot(range(epochs), test_losses, label="Test loss")
    plt.plot(range(epochs), eval_losses, label="Eval loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.savefig("loss_curve.png")
    plt.close()



def plot_network_errors(trafo_id, errors, colab=False):

    #get data
    raw_data,_ = read_raw_network_data(trafo_id, colab)
    raw_data = fill_data_simple_homogeneous(raw_data)
    data = preprocess(raw_data)

    #transform raw_data["nodes_static_data"] to a list ordered by node_id and put x_y coordinates to the list
    nodes_static_data = data["nodes_static_data"]
    nodes_static_data = nodes_static_data.sort_values(by=['node_id'])
    nodes_static_data = nodes_static_data.reset_index(drop=True)
    nodes_coords = nodes_static_data[["x_y"]]
    nodes_coords = nodes_coords.values.tolist()

    #this is it now ['(530670.5350000001, 153984.66459999979)'] now we need to split the string and convert to float
    nodes_coords = [x[0].split(",") for x in nodes_coords]
    nodes_coords = [[float(x[0][1:]),float(x[1][:-1])] for x in nodes_coords]

    # get the nodes that are not pmo
    df_unique_nodes = nodes_static_data.drop_duplicates(subset='node_id')
    result_df = df_unique_nodes[df_unique_nodes['aclass_id'] != 'PMO']
    not_pmo_nodes = result_df['node_id'].tolist()

    #create the graph
    G = nx.from_pandas_edgelist(data['edges_static_data'],source='from_node_id',target='to_node_id')

    #for each first dim change the second dim on index from not_pmo_nodes to nan
    for i in range(len(errors)):
        for j in range(len(errors[i])):
            if j in not_pmo_nodes:
                errors[i][j] = np.nan

    # Plot the graph
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter'}]])
    color_scale = [[0, 'green'], [0.5, 'darkorange'], [1, 'darkred']]
    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = nodes_coords[edge[0]]
        x1, y1 = nodes_coords[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    # Create edges trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_x = [nodes_coords[k][0] for k in range(len(nodes_coords))]
    node_y = [nodes_coords[k][1] for k in range(len(nodes_coords))]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale=color_scale,
            size=10,
            colorbar=dict(
                thickness=15,
                xanchor='left',
                titleside='right'
            )
        )
    )
    # Create layout
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800,
        width=800,
    )
    # Add initial frame
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    # Add frames for animation
    frames = []
    for frame_idx in range(len(errors)):
        # Update node color and text for each frame
        node_trace.marker.color = errors[frame_idx]

        node_trace.marker.cmin = 0
        node_trace.marker.cmax = 3
        
        node_text = []
        for node in range(len(nodes_coords)):
            node_text.append(f'Node ID: {node}<br>Error: {errors[frame_idx][node]:.2f}')
        node_trace.text = node_text
        # Add the node and edge traces to the frames
        frames.append(go.Frame(data=[edge_trace, node_trace], name=f'Frame {frame_idx}'))
    # Update layout for animation
    layout.updatemenus = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 100},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]
    layout.sliders = [
        {
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'steps': [
                {
                    'args': [
                        [f'{frame_idx}'],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f'{frame_idx}',
                    'method': 'animate'
                } for frame_idx in range(len(errors))
            ]
        }
    ]
    fig.frames = frames
    fig.update_layout(layout)
    fig.show()

