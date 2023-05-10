import networkx as nx


class graph_generator:

    def __init__(self,
                 num_of_enc_layer: int,
                 num_of_enc_head: int,
                 num_of_dec_layer: int,
                 num_of_dec_head: int,
                 is_encoder: bool,
                 is_self: bool):
        self.num_of_enc_layer = num_of_enc_layer
        self.num_of_enc_head = num_of_enc_head
        self.num_of_dec_layer = num_of_dec_layer
        self.num_of_dec_head = num_of_dec_head
        self.is_encoder = is_encoder
        self.is_self = is_self

    def gen_head(self, input_graph, prefix, layer_idx, head_idx, input_node, output_node, input_enc):
        QKt_head_nm = prefix + str(layer_idx) + '_QKt_' + str(head_idx)
        input_graph.add_node(QKt_head_nm, label=QKt_head_nm)
        input_graph.add_edge(input_node, QKt_head_nm)

        matmul_1_head_nm = prefix + str(layer_idx) + '_matmul_1_' + str(head_idx)
        input_graph.add_node(matmul_1_head_nm, label=matmul_1_head_nm)
        matmul_2_head_nm = prefix + str(layer_idx) + '_matmul_2_' + str(head_idx)
        input_graph.add_node(matmul_2_head_nm, label=matmul_2_head_nm)

        if "D_enc_" in prefix and not self.is_self:
            input_graph.add_edge(input_enc, QKt_head_nm)
            input_graph.add_edge(input_enc, matmul_1_head_nm)
        else:
            input_graph.add_edge(input_node, matmul_1_head_nm)

        input_graph.add_edge(QKt_head_nm, matmul_1_head_nm)
        input_graph.add_edge(matmul_1_head_nm, matmul_2_head_nm)

        VO_head_nm = prefix + str(layer_idx) + '_VO_' + str(head_idx)
        input_graph.add_node(VO_head_nm, label=VO_head_nm)
        input_graph.add_edge(VO_head_nm, matmul_2_head_nm)
        # sum all heads
        input_graph.add_edge(matmul_2_head_nm, output_node)

        return input_graph

    def gen_block(self, input_graph, prefix, current_layer, num_layer, num_head):
        # add input/output nodes as many as given layer+1
        input_nm = prefix + str(current_layer)
        input_graph.add_node(input_nm, label=input_nm)
        # connect previous sum to current input
        if current_layer > 0:
            input_graph.add_edge(prefix + str(current_layer - 1) + '_add', input_nm)
        # final input is output; no more graph
        if current_layer == num_layer:
            return input_graph
        else:
            # add node for sum of  all heads
            sum_nm = prefix + str(current_layer) + '_add'
            input_graph.add_node(sum_nm, label=sum_nm)

            if "E_self" not in prefix and not self.is_self:
                encdec_nm = "D_enc_" + str(current_layer)
                input_graph.add_node(encdec_nm, label=encdec_nm)
                # add QKt, VO, Mask(opt.), matmul nodes between layers as many as heads
                for head in range(num_head):
                    # seq2seq : dec-self
                    input_graph = self.gen_head(input_graph=input_graph,
                                                prefix=prefix,
                                                layer_idx=current_layer,
                                                head_idx=head,
                                                input_node=input_nm,
                                                output_node=encdec_nm,
                                                input_enc="")
                    # seq2seq : connect dec-enc
                    input_graph = self.gen_head(input_graph=input_graph,
                                                prefix="D_enc_",
                                                layer_idx=current_layer,
                                                head_idx=head,
                                                input_node=encdec_nm,
                                                output_node=sum_nm,
                                                input_enc="E_self_" + str(self.num_of_enc_layer))
                # skip connection
                input_graph.add_edge(input_nm, encdec_nm)
                input_graph.add_edge(encdec_nm, sum_nm)
            else:
                # add QKt, VO, Mask(opt.), matmul nodes between layers as many as heads
                for head in range(num_head):
                    input_graph = self.gen_head(input_graph=input_graph,
                                                prefix=prefix,
                                                layer_idx=current_layer,
                                                head_idx=head,
                                                input_node=input_nm,
                                                output_node=sum_nm,
                                                input_enc="")
                # skip connection
                input_graph.add_edge(input_nm, sum_nm)
            return input_graph

    def gen_graph(self):
        G = nx.DiGraph()
        if self.is_encoder:
            for layer in range(self.num_of_enc_layer + 1):
                G = self.gen_block(input_graph=G,
                                   prefix="E_self_",
                                   current_layer=layer,
                                   num_layer=self.num_of_enc_layer,
                                   num_head=self.num_of_enc_head)
        else:
            if self.is_self:
                for layer in range(self.num_of_dec_layer + 1):
                    G = self.gen_block(input_graph=G,
                                       prefix="D_self_",
                                       current_layer=layer,
                                       num_layer=self.num_of_dec_layer,
                                       num_head=self.num_of_dec_head)
            else:
                # seq2seq
                # for encoder
                for layer in range(self.num_of_enc_layer + 1):
                    G = self.gen_block(input_graph=G,
                                       prefix="E_self_",
                                       current_layer=layer,
                                       num_layer=self.num_of_enc_layer,
                                       num_head=self.num_of_enc_head)
                # for decoder
                for layer in range(self.num_of_dec_layer + 1):
                    G = self.gen_block(input_graph=G,
                                       prefix="D_self_",
                                       current_layer=layer,
                                       num_layer=self.num_of_dec_layer,
                                       num_head=self.num_of_dec_head)
        return G

