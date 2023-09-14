import torch
import numpy as np
from graph_bridges.models.networks.graphs_networks.edp_gnn import EdgeDensePredictionGraphScoreNetwork

class GraphScorePerTime:
    """
    The estimator calculates the cost for a whole path
    """
    name_ = "graph_score_estimator"
    score_model : EdgeDensePredictionGraphScoreNetwork

    def __init__(self, **kwargs):
        super(GraphScorePerTime, self).__init__(**kwargs)
        self.number_of_spins = kwargs.get("number_of_spins")
        self.time_embedding_dim = kwargs.get("time_embedding_dim", 10)
        self.gnn_hidden_num_list = kwargs.get("gnn_hidden_num_list")
        self.max_node_num =  kwargs.get("max_node_num")
        self.number_of_classes = kwargs.get("number_of_classes",6)
        self.feature_nums = kwargs.get("feature_nums")
        self.channel_num_list = kwargs.get("channel_num_list")
        self.graph_func_name = kwargs.get("graph_func_name")

        dev = torch.device("cpu")
        self.score_model = self.define_deep_models(dev)

    def define_deep_models(self,dev):

        def gnn_model_func(**gnn_params):
            merged_params = self.obtain_parameters()
            merged_params.update(gnn_params)
            return NAME_TO_CLASS[self.graph_func_name](**merged_params).to(dev)

        score_model = EdgeDensePredictionGraphScoreNetwork(feature_num_list=self.feature_nums,
                                                           channel_num_list=self.channel_num_list,
                                                           max_node_number=self.max_node_num,
                                                           gnn_hidden_num_list=self.gnn_hidden_num_list,
                                                           gnn_module_func=gnn_model_func,
                                                           dev=dev,
                                                           num_classes=self.number_of_classes).to(dev)

        return score_model

    def to(self,device):
        gnn_list =  []
        for graph_func in self.score_model.gnn_list:
            graph_func.to(device)
        #self.score_model = gnn_list
        self.score_model = self.score_model.to(device)
        self.score_model.mask = self.score_model.mask.to(device)
        return self

    def init_weights(self):
        return None

    def forward_states_and_times(self,
                                 states,
                                 times):
        """
        :param states:
        :param times:
        :return:
        """

        time_embbedings = get_timestep_embedding(times.squeeze(),
                                                 time_embedding_dim=self.time_embedding_dim)

        time_embbedings = time_embbedings.unsqueeze(-1)

        number_of_nodes = int(np.sqrt(self.number_of_spins))
        number_of_classes = self.number_of_classes

        batch_and_time_size, number_of_entries = states.shape
        states = states.view(batch_and_time_size,
                                 number_of_nodes,
                                 number_of_nodes)

        states = states.repeat((number_of_classes, 1, 1))
        node_flags = torch.ones(batch_and_time_size * number_of_classes, number_of_nodes).to(time_embbedings.device)
        self.score_model.to(time_embbedings.device)

        x = torch.zeros(batch_and_time_size * number_of_classes, number_of_nodes, 1).to(time_embbedings.device)
        time_embbedings = time_embbedings.repeat((1,1,number_of_nodes))
        time_embbedings = time_embbedings.permute(0,2,1)

        x = torch.cat([x,time_embbedings],dim=-1)
        score_ = self.score_model(x=x,
                                  adjs=states,
                                  node_flags=node_flags)
        score_ = softplus(score_)
        return score_.reshape(batch_and_time_size,number_of_entries)

    @classmethod
    def get_parameters(self) -> dict:
        kwargs = super().get_parameters()
        time_embedding_dim = kwargs.get('time_embedding_dim')

        # FEATURES SIZE -----------------------------------------------------
        features_vectors_size = 1
        in_features = features_vectors_size + time_embedding_dim - 1
        feature_nums = [2]
        feature_nums = [in_features + 1] + feature_nums
        # FEATURES SIZE -----------------------------------------------------

        kwargs.update({'dropout_p': 0.0,
                       'gnn_hidden_num_list': [2],
                       'feature_nums': feature_nums,
                       'channel_num_list': [2],
                       'graph_func_name': 'gin',
                       'max_node_num':10,
                       "number_of_classes":1,
                       'use_norm_layers': False})

        return kwargs