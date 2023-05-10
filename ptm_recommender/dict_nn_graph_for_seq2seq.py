import torch
from ptm_recommender.dict_nn_graph import dict_nn_graph


# for seq2seq (bart, t5)
SEQ_QUERY_NM = ["encoder.block.{n}.layer.0.SelfAttention.q.weight",
                "encoder.layers.{n}.self_attn.q_proj.weight",
                "decoder.block.{n}.layer.0.SelfAttention.q.weight",
                "decoder.layers.{n}.self_attn.q_proj.weight",
                "decoder.block.{n}.layer.1.EncDecAttention.q.weight",
                "decoder.layers.{n}.encoder_attn.q_proj.weight"]

SEQ_KEY_NM = ["encoder.block.{n}.layer.0.SelfAttention.k.weight",
              "encoder.layers.{n}.self_attn.k_proj.weight",
              "decoder.block.{n}.layer.0.SelfAttention.k.weight",
              "decoder.layers.{n}.self_attn.k_proj.weight",
              "decoder.block.{n}.layer.1.EncDecAttention.k.weight",
              "decoder.layers.{n}.encoder_attn.k_proj.weight"]

SEQ_VALUE_NM = ["encoder.block.{n}.layer.0.SelfAttention.v.weight",
                "encoder.layers.{n}.self_attn.v_proj.weight",
                "decoder.block.{n}.layer.0.SelfAttention.v.weight",
                "decoder.layers.{n}.self_attn.v_proj.weight",
                "decoder.block.{n}.layer.1.EncDecAttention.v.weight",
                "decoder.layers.{n}.encoder_attn.v_proj.weight"]

SEQ_OUTPUT_NM = ["encoder.block.{n}.layer.0.SelfAttention.o.weight",
                 "encoder.layers.{n}.self_attn.out_proj.weight",
                 "decoder.block.{n}.layer.0.SelfAttention.o.weight",
                 "decoder.layers.{n}.self_attn.out_proj.weight",
                 "decoder.block.{n}.layer.1.EncDecAttention.o.weight",
                 "decoder.layers.{n}.encoder_attn.out_proj.weight"]


class dict_nn_graph_for_seq2seq(dict_nn_graph):

    def __init__(self, model, config):
        super().__init__(model, config, is_encoder=False)

    def get_model_dict(self):
        # For t5, bart, and mbart, encoder and decoder layers have the same
        #   hidden_size, num_attention_heads, num_hidden_layers
        d = self.config.hidden_size
        h = self.config.num_attention_heads
        m = self.config.num_hidden_layers
        layer = 0

        # This loop assumes that the model consecutively provides list of parameters by layer
        nodes_nm_in_model = [n for n, _ in self.model.named_parameters()]
        param_dict = dict(self.model.named_parameters())

        while layer < m:
            for n_idx in range(len(nodes_nm_in_model)):

                nm = nodes_nm_in_model[n_idx]

                if self.is_in_list(nm, layer, SEQ_QUERY_NM):
                    self.weight_query = torch.tensor(param_dict[nm])
                elif self.is_in_list(nm, layer, SEQ_KEY_NM):
                    self.weight_key = torch.tensor(param_dict[nm])
                elif self.is_in_list(nm, layer, SEQ_VALUE_NM):
                    self.weight_value = torch.tensor(param_dict[nm])
                elif self.is_in_list(nm, layer, SEQ_OUTPUT_NM):
                    self.weight_output = torch.tensor(param_dict[nm])
                else:
                    continue  # continue for loop

                if nm.lower().startswith('encoder'):
                    prefix = "E_self_"
                elif nm.lower().startswith('decoder'):
                    if "enc" in nm.lower():
                        prefix = "D_enc_"
                    else:
                        prefix = "D_self_"

                # once weights are collected in the layer compute eigenvalues and remove them for the next layer
                if self.isNone():
                    for idx in range(h):
                        self.getFeature(prefix, layer, idx, d, h)
                    self.letNone()

            layer = layer + 1
            self.letNone()

        return self.inner_model_dict
