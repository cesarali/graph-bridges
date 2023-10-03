import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import warnings

#warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)


options = {
    'node_size': 2,
    'edge_color': 'black',
    'linewidths': 1,
    'width': 0.5
}

CMAP = cm.jet


def plot_graphs_list2(graphs,
                      energy=None,
                      node_energy_list=None,
                      title='title',
                      max_num=16,
                      save_dir=None):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(np.sqrt(max_num))
    figure = plt.figure()

    for i in range(max_num):
        idx = i * (batch_size // max_num)
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        #l = G.number_of_selfloops()
        l = sum([1 for n in G.nodes() if G.has_edge(n, n)])

        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        if energy is not None:
            title_str += f'\n en={energy[idx]:.1e}'

        if node_energy_list is not None:
            node_energy = node_energy_list[idx]
            title_str += f'\n {np.std(node_energy):.1e}'
            nx.draw(G, with_labels=False, node_color=node_energy, cmap=cm.jet, **options)
        else:
            # print(nx.get_node_attributes(G, 'feature'))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    figure.suptitle(title)
    if save_dir is not None:
        figure.savefig(save_dir)
    else:
        plt.show()