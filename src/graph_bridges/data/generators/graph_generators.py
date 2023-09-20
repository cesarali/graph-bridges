import json
import pickle
import numpy as np
import networkx as nx

def n_community(num_communities, max_nodes, p_inter=0.05):
    assert num_communities > 1

    one_community_size = max_nodes // num_communities
    c_sizes = [one_community_size] * num_communities
    total_nodes = one_community_size * num_communities

    """ 
    here we calculate `p_make_a_bridge` so that `p_inter = \mathbb{E}(Number_of_bridge_edges) / Total_number_of_nodes `

    To make it more clear: 
    let `M = num_communities` and `N = one_community_size`, then

    ```
    p_inter
    = \mathbb{E}(Number_of_bridge_edges) / Total_number_of_nodes
    = (p_make_a_bridge * C_M^2 * N^2) / (MN)  # see the code below for this derivation
    = p_make_a_bridge * (M-1) * N / 2
    ```

    so we have:
    """
    p_make_a_bridge = p_inter * 2 / ((num_communities - 1) * one_community_size)

    print(num_communities, total_nodes, end=' ')
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]

    G = nx.disjoint_union_all(graphs)
    # communities = list(nx.connected_component_subgraphs(G))
    communities = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    add_edge = 0
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):  # loop for C_M^2 times
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:  # loop for N times
                for n2 in nodes2:  # loop for N times
                    if np.random.rand() < p_make_a_bridge:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
                        add_edge += 1
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
                add_edge += 1

    # print('connected comp: ', len(list(nx.connected_component_subgraphs(G))),
    print('connected comp: ', len([G.subgraph(c).copy() for c in nx.connected_components(G)]),
          'add edges: ', add_edge)
    print(G.number_of_edges())
    return G


NAME_TO_NX_GENERATOR = {
    'community': n_community,
    'grid': nx.generators.grid_2d_graph,  # grid_2d_graph(m, n, periodic=False, create_using=None)
    'gnp': nx.generators.fast_gnp_random_graph,  # fast_gnp_random_graph(n, p, seed=None, directed=False)
    'ba': nx.generators.barabasi_albert_graph,  # barabasi_albert_graph(n, m, seed=None)
    'pow_law': lambda **kwargs: nx.configuration_model(nx.generators.random_powerlaw_tree_sequence(**kwargs,
                                                                                                   gamma=3,
                                                                                                   tries=2000)),
    'except_deg': lambda **kwargs: nx.expected_degree_graph(**kwargs, selfloops=False),
    'cycle': nx.cycle_graph,
    'c_l': nx.circular_ladder_graph,
    'lobster': nx.random_lobster
    # 'ego': nx.generators.ego_graph  # ego_graph(G, n, radius=1, center=True, undirected=False, distance=None)
}

class GraphGenerator:
    def __init__(self, graph_type='grid', possible_params_dict=None, corrupt_func=None):
        if possible_params_dict is None:
            possible_params_dict = {}
        assert isinstance(possible_params_dict, dict)
        self.count = {k: 0 for k in possible_params_dict}
        self.possible_params = possible_params_dict
        self.corrupt_func = corrupt_func
        self.nx_generator = NAME_TO_NX_GENERATOR[graph_type]

    def __call__(self):
        params = {}
        for k, v_list in self.possible_params.items():
            params[k] = np.random.choice(v_list)
        graph = self.nx_generator(**params)
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        if self.corrupt_func is not None:
            graph = self.corrupt_func(self.corrupt_func)
        return graph

def gen_graph_list(graph_type='grid', possible_params_dict=None, corrupt_func=None, length=1024, max_node=None, min_node=None,**kwargs):
    params = locals()
    graph_generator = GraphGenerator(graph_type=graph_type,
                                     possible_params_dict=possible_params_dict,
                                     corrupt_func=corrupt_func)
    graph_list = []
    i = 0
    max_N = 0
    while i < length:
        graph = graph_generator()
        if max_node is not None and graph.number_of_nodes() > max_node:
            continue
        if min_node is not None and graph.number_of_nodes() < min_node:
            continue
        print(i, graph.number_of_nodes(), graph.number_of_edges())
        max_N = max(max_N, graph.number_of_nodes())
        if graph.number_of_nodes() <= 1:
            continue
        graph_list.append(graph)
        i += 1
    print(max_N)
    return graph_list

if __name__=="__main__":
    file_name = 'community_small'

    res_graph_list = gen_graph_list(graph_type='community', possible_params_dict={
        'num_communities': [2],
        'max_nodes': [10],
    }, corrupt_func=None, length=10,max_node=10 ,min_node=10)

    res_graph_list = gen_graph_list(graph_type='lobster',
                                    possible_params_dict={
                                        'n': np.arange(5, 16).tolist(),
                                        'p1': [0.7],
                                        'p2': [0.5]
                                    },
                                    corrupt_func=None,
                                    length=10,
                                    max_node=10,
                                    min_node=10)
    print(res_graph_list)