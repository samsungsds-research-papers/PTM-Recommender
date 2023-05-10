import networkx as nx


def node_iter(G):
    if nx.__version__ < '2.0':
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if nx.__version__ > '2.1':
        return G.nodes
    else:
        return G.node
