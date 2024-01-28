
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt


def plot_graph_2d(graph, nodes_shape, plot_weights=True, plot_terminals=True, font_size=7):

    X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: v for i, v in enumerate(aux)}
    positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
    positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

    nxgraph = graph.get_nx_graph()
    if not plot_terminals:
        nxgraph.remove_nodes_from(['s', 't'])

    plt.clf()
    nx.draw(nxgraph, pos=positions)

    if plot_weights:
        edge_labels = {}
        for u, v, d in nxgraph.edges(data=True):
            edge_labels[(u, v)] = d['weight']
        nx.draw_networkx_edge_labels(nxgraph,
                                     pos=positions,
                                     edge_labels=edge_labels,
                                     label_pos=0.3,
                                     font_size=font_size)

    plt.axis('equal')
    plt.show()


def plot_graph_3d(graph, nodes_shape, plot_terminal=True, plot_weights=True, font_size=7):

    w_h = nodes_shape[1] * nodes_shape[2]
    X, Y = np.mgrid[:nodes_shape[1], :nodes_shape[2]]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: v for i, v in enumerate(aux)}

    for i in range(1, nodes_shape[0]):
        for j in range(w_h):
            positions[w_h * i + j] = [positions[j][0] + 0.3 * i, positions[j][1] + 0.2 * i]

    positions['s'] = np.array([-1, nodes_shape[1] / 2.0 - 0.5])
    positions['t'] = np.array([nodes_shape[2] + 0.2 * nodes_shape[0], nodes_shape[1] / 2.0 - 0.5])

    nxg = graph.get_nx_graph()
    if not plot_terminal:
        nxg.remove_nodes_from(['s', 't'])

    nx.draw(nxg, pos=positions)
    nx.draw_networkx_labels(nxg, pos=positions)
    if plot_weights:
        edge_labels = dict([((u, v), d['weight']) for u, v, d in nxg.edges(data=True)])
        nx.draw_networkx_edge_labels(nxg,
                                     pos=positions,
                                     edge_labels=edge_labels,
                                     label_pos=0.3,
                                     font_size=font_size)
    plt.axis('equal')
    plt.show()

"""
This file contains a list of examples with different layouts that can be
obtained using the ``add_grid_edges`` method.
"""

import numpy as np
import maxflow


# # Standard 4-connected grid
# g = maxflow.Graph[int]()
# nodeids = g.add_grid_nodes((5, 5))
# g.add_grid_edges(nodeids, 1)
# # Equivalent to
# # structure = maxflow.vonNeumann_structure(ndim=2, directed=False)
# # g.add_grid_edges(nodeids, 1,
# #                  structure=structure,
# #                  symmetric=False)
# plot_graph_2d(g, nodeids.shape, plot_terminals=False)

# 8-connected grid, ignore two nodes
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5, 5))
structure = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]])
# Also structure = maxflow.moore_structure(ndim=2, directed=True)
# Nodeid -1 is ignored when adding edges
# nodeids[1, 1] = -1
# nodeids[3, 2] = -1
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=True)
plot_graph_2d(g, nodeids.shape, plot_terminals=False)
import pdb;pdb.set_trace()
# 24-connected 5x5 neighborhood
# g = maxflow.Graph[int]()
# nodeids = g.add_grid_nodes((5, 5))
# structure = np.array([[1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1],
#                       [1, 1, 0, 1, 1],
#                       [1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1]])
# g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
# plot_graph_2d(g, nodeids.shape, plot_terminals=False, plot_weights=False)

# # Diagonal, not symmetric
# g = maxflow.Graph[int]()
# nodeids = g.add_grid_nodes((5, 5))
# structure = np.array([[0, 0, 0],
#                       [0, 0, 0],
#                       [0, 0, 1]])
# g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
# plot_graph_2d(g, nodeids.shape, plot_terminals=False)

# # Central node connected to every other node
# g = maxflow.Graph[int]()
# nodeids = g.add_grid_nodes((5, 5)).ravel()

# central_node = nodeids[12]
# rest_of_nodes = np.hstack([nodeids[:12], nodeids[13:]])

# nodeids = np.empty((2, 24), dtype=np.int_)
# nodeids[0] = central_node
# nodeids[1] = rest_of_nodes

# structure = np.array([[0, 0, 0],
#                       [0, 0, 0],
#                       [0, 1, 0]])
# g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
# plot_graph_2d(g, (5, 5), plot_terminals=False)