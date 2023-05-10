import torch
from ptm_recommender.dict_nn_graph import dict_nn_graph


class dict_nn_graph_for_albert(dict_nn_graph):

    def __init__(self, model, config):
        super().__init__(model, config, is_encoder=True)

    def get_model_dict(self):
        d = self.config.hidden_size
        h = self.config.num_attention_heads
        m = self.config.num_hidden_layers
        layer = 0

        # This loop assumes that the model consecutively provides list of parameters by layer
        nodes_nm_in_model = [n for n, _ in self.model.named_parameters()]
        param_dict = dict(self.model.named_parameters())

        for n_idx in range(len(nodes_nm_in_model)):
            nm = nodes_nm_in_model[n_idx]
            # get weights for given "layer"
            if all(x in nm for x in ['attention', 'weight']) and "LayerNorm" not in nm:
                if "query" in nm:
                    self.weight_query = torch.tensor(param_dict[nm])
                elif "key" in nm:
                    self.weight_key = torch.tensor(param_dict[nm])
                elif "value" in nm:
                    self.weight_value = torch.tensor(param_dict[nm])
                elif "dense" in nm:
                    self.weight_output = torch.tensor(param_dict[nm])
            else:
                continue  # continue for loop

        # get shared weights
        for layer in range(m):
            for idx in range(h):
                self.getFeature("E_self_", layer, idx, d, h)

        return self.inner_model_dict
