import torch


# for encoder only (w/o albert, longformer, funnel)
ENC_QUERY_NM = ["encoder.layer.{n}.attention.self.query.weight",
                "transformer.layer.{n}.attention.q_lin.weight",
                "encoder.layer.{n}.attention.self.query.weight",
                "attentions.{n}.q_lin.weight"]

ENC_KEY_NM = ["encoder.layer.{n}.attention.self.key.weight",
              "transformer.layer.{n}.attention.k_lin.weight",
              "encoder.layer.{n}.attention.self.key.weight",
              "attentions.{n}.k_lin.weight"]

ENC_VALUE_NM = ["encoder.layer.{n}.attention.self.value.weight",
                "transformer.layer.{n}.attention.v_lin.weight",
                "encoder.layer.{n}.attention.self.value.weight",
                "attentions.{n}.v_lin.weight"]

ENC_OUTPUT_NM=["encoder.layer.{n}.attention.output.dense.weight",
               "transformer.layer.{n}.attention.out_lin.weight",
               "encoder.layer.{n}.attention.output.dense.weight",
               "attentions.{n}.out_lin.weight"]

# for decoder only (gpt based model)
DEC_QKV_NM = ["h.{n}.attn.c_attn.weight"]
DEC_OUTPUT_NM = ["h.{n}.attn.c_proj.weight"]

class dict_nn_graph:
    def __init__(self, model, config, is_encoder):

        self.model = model
        self.config = config
        self.weight_query = None
        self.weight_key = None
        self.weight_value = None
        self.weight_output = None
        self.inner_model_dict = {}
        self.is_encoder = is_encoder

        # Due to A100 CPU issue in x.cloud, reduce the number of threads
        torch.set_num_threads(2)

    def get_eigvals_low_rank_matrix(self, mat_a, mat_b):
        # This is for computing nonzero eigenvalues of AB
        # Exploit the fact that nonzero eigenvalues of AB are equal to those of BA
        # The size of rows of A = The size of columns of B = M
        # The size of columns of A = The size of rows of B = N
        #   M >> N
        # This is very useful for low-rank square matrix which we know what A,B are
        # Return flatten coefficients of complex eigenvalues
        # e.g., e = [2 + i, -1 + 3i, 2 - 2i] where a,b in R
        #       e -> [2, 1, -1, 3, 2, -2]

        mat_ba = torch.matmul(mat_b, mat_a)

        # ------ for torch.eig() ------
        #     e = torch.linalg.eigvals(BA)
        #     e = torch.view_as_real(e)
        # ------ for torch.linalg.eigvals() ------
        
        eig_vals, _ = torch.linalg.eig(mat_ba)
        # sort eig_vals
        e = self.sort_eigval_set(eig_vals,
                                 descend=True,
                                 primary="real",
                                 secondary="image")
        return torch.flatten(e)

    @staticmethod
    def sort_eigval_set(c_array, descend=True, primary="real", secondary="image"):
        # This function sorts complex eigenvalue under two rules given order
        # e.g.) primary = "real", secondary = "image"
        #       primary = "abs", secondary = "angle"
        #
        # descend(option) = True or False
        #
        if len(c_array.shape) > 1:
            c_array = torch.flatten(c_array)

        c_real_imag = torch.view_as_real(c_array)
        c_abs = torch.reshape(c_array.abs(), (len(c_array), 1))
        c_angle = torch.reshape(c_array.angle(), (len(c_array), 1))
        c_table = torch.cat((c_real_imag, c_abs, c_angle), 1)

        # real: 0 , image: 1, abs: 2, angle: 3
        sort_rule_dict = {'real': 0, 'image': 1, 'abs': 2, 'angle': 3}
        primary_rule = sort_rule_dict[primary]
        secondary_rule = sort_rule_dict[secondary]

        c_table = c_table[c_table[:, secondary_rule].sort(descending=descend)[1]]
        c_table = c_table[c_table[:, primary_rule].sort(descending=descend)[1]]

        return c_table[:, :2]

    def isNone(self):
        return None not in (self.weight_query, self.weight_key, self.weight_value, self.weight_output)

    def letNone(self):
        self.weight_query = None
        self.weight_key = None
        self.weight_value = None
        self.weight_output = None

    def getFeature(self, prefix: str, layer: int, idx: int, d: int, h: int):
        s = (d // h) * idx
        e = (d // h) * (idx + 1)
        eig_QKt_h = self.get_eigvals_low_rank_matrix(self.weight_query[:, s:e],
                                                     torch.transpose(self.weight_key[:, s:e], 0, 1))
        eig_VO_h = self.get_eigvals_low_rank_matrix(self.weight_value[:, s:e], self.weight_output[s:e, :])

        if prefix == 'D_self_':  # for masking feature
            eig_QKt_h = torch.cat((torch.zeros(128), eig_QKt_h), 0)
            eig_VO_h = torch.cat((torch.zeros(128), eig_VO_h), 0)
        else:
            eig_QKt_h = torch.cat((eig_QKt_h, torch.zeros(128)), 0)
            eig_VO_h = torch.cat((eig_VO_h, torch.zeros(128)), 0)

        self.inner_model_dict[prefix + str(layer) + "_QKt_" + str(idx)] = eig_QKt_h
        self.inner_model_dict[prefix + str(layer) + "_VO_" + str(idx)] = eig_VO_h

    @staticmethod
    def is_in_list(nm, layer, nm_list):
        return any(ext.replace('{n}', str(layer)) in nm for ext in nm_list)

    def get_model_dict(self):
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

                # for autoencoding
                # get weights for given "layer"
                if self.is_encoder:
                    prefix = "E_self_"
                    if self.is_in_list(nm, layer, ENC_QUERY_NM):
                        self.weight_query = torch.tensor(param_dict[nm])
                    elif self.is_in_list(nm, layer, ENC_KEY_NM):
                        self.weight_key = torch.tensor(param_dict[nm])
                    elif self.is_in_list(nm, layer, ENC_VALUE_NM):
                        self.weight_value = torch.tensor(param_dict[nm])
                    elif self.is_in_list(nm, layer, ENC_OUTPUT_NM):
                        self.weight_output = torch.tensor(param_dict[nm])
                    else:
                        continue
                # for autoregressive
                # get weights for given "layer"
                else:
                    prefix = "D_self_"
                    if self.is_in_list(nm, layer, DEC_QKV_NM):
                        self.weight_query = torch.tensor(param_dict[nm])[:, 0:d]
                        self.weight_key = torch.tensor(param_dict[nm])[:, d:d * 2]
                        self.weight_value = torch.tensor(param_dict[nm])[:, d * 2:d * 3]
                    elif self.is_in_list(nm, layer, DEC_OUTPUT_NM):
                        self.weight_output = torch.tensor(param_dict[nm])
                    else:
                        continue  # continue for loop

                # once weights are collected in the layer, compute eigenvalues and remove them for the next layer
                if self.isNone():
                    for idx in range(h):
                        self.getFeature(prefix, layer, idx, d, h)
                    self.letNone()

            layer = layer + 1
            self.letNone()

        return self.inner_model_dict
