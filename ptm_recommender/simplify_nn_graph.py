import torch
from torchviz import make_dot
import pydotplus
import copy
import networkx as nx
from ptm_recommender.graph_models import graph_util as util


class simplify_nn_graph:

    def __init__(self, model, config, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

    @staticmethod
    def remove_node(input_graph):
        inner_graph = copy.deepcopy(input_graph)
        # remove node whose label contains one of items in the below list
        rm_list = ['bias', 'intermediate', 'ffn', 'layer_norm', 'LayerNorm.weight', 'LayerNorm.bias']
        flag_while = 1
        while flag_while:
            for node in inner_graph.nodes:
                flag_while = 0
                # label이 NativeLayerNormBackward에서 다른 노드의 정보를 가진 경우 (e.g., bias) 삭제됨
                node_label = util.node_iter(inner_graph)[node]['label'].split("\n")[0]
                if any(ext in node_label for ext in rm_list):
                    successor = list(inner_graph.successors(node))[0]
                    inner_graph.remove_edge(node, successor)
                    inner_graph.remove_node(node)
                    flag_while = 1
                    break
                # Remove node whose predecessor does not exist
                # unless its label contains "NativeLayerNormBackward" (node connecting layers) or "attention" (weight)
                if "NativeLayerNormBackward" not in node_label and "attention" not in node_label and len(
                        list(inner_graph.predecessors(node))) == 0:
                    successor = list(inner_graph.successors(node))[0]
                    inner_graph.remove_edge(node, successor)
                    inner_graph.remove_node(node)
                    flag_while = 1
                    break
        return inner_graph

    @staticmethod
    # node의 in / out이 1개씩 있는 단순한 구조일 경우 제거하고 predecessor와 successor를 연결
    def skip_node(input_graph):
        inner_graph = copy.deepcopy(input_graph)
        total_node = [node for node in inner_graph]
        non_removable_node = [node for node in inner_graph if
                              not (len(list(inner_graph.predecessors(node))) == 1 and
                                   len(list(inner_graph.successors(node))) == 1)]
        while len(total_node) > len(non_removable_node):
            for node in inner_graph.nodes:
                if len(list(inner_graph.predecessors(node))) == 1 and \
                        len(list(inner_graph.successors(node))) == 1:
                    predecessor = list(inner_graph.predecessors(node))[0]
                    successor = list(inner_graph.successors(node))[0]
                    inner_graph.remove_edge(predecessor, node)
                    inner_graph.remove_edge(node, successor)
                    inner_graph.remove_node(node)
                    inner_graph.add_edge(predecessor, successor)
                    break
            total_node = [node for node in inner_graph]
            non_removable_node = [node for node in inner_graph if
                                  not (len(list(inner_graph.predecessors(node))) == 1 and
                                       len(list(inner_graph.successors(node))) == 1)]
        return inner_graph

    @staticmethod
    # Attention weight node (which has no predecessor and has only one successor) replace its successor
    def replace_node(input_graph):
        inner_graph = copy.deepcopy(input_graph)
        flag_while = 1
        while flag_while:
            for node in inner_graph.nodes:
                flag_while = 0
                if "attention" in util.node_dict(inner_graph)[node]['label'].split("\n")[0] \
                        and len(list(inner_graph.predecessors(node))) == 0:
                    # Assuming there is unique successor for each node
                    successor = list(inner_graph.successors(node))[0]
                    # successors of node's successor
                    successor_successor = list(inner_graph.successors(successor))[0]
                    successor_another_predecessors = list(inner_graph.predecessors(successor))
                    successor_another_predecessors.remove(node)
                    successor_another_predecessor = successor_another_predecessors[0]
                    inner_graph.remove_edge(node, successor)
                    inner_graph.remove_edge(successor_another_predecessor, successor)
                    inner_graph.remove_edge(successor, successor_successor)
                    inner_graph.remove_node(successor)
                    inner_graph.add_edge(successor_another_predecessor, node)
                    inner_graph.add_edge(node, successor_successor)
                    flag_while = 1
                    break
        return inner_graph

    @staticmethod
    # This creates nodes as many as attention heads under "NativeLayerNormBackward"
    # to split weight matrix operations
    def add_nodes_for_each_head(input_graph, head: int):
        inner_graph = copy.deepcopy(input_graph)
        node_list = [node for node in inner_graph.nodes]
        to_be_removed_list = []
        # Add nodes
        for node in node_list:
            start_label = util.node_iter(inner_graph)[node]['label'].split("\n")[0]
            if "NativeLayerNormBackward" in start_label:
                start_node = node
                successors = list(inner_graph.successors(node))
                end_node = None
                layer = None
                node_add_heads = None
                for successor in successors:
                    successor_label = util.node_iter(inner_graph)[successor]['label'].split("\n")[0]
                    if "AddBackward0" in successor_label:
                        end_node = successor
                    else:
                        to_be_removed_list.append(successor)
                        layer = successor_label.split(".")[2]
                for h in range(head):
                    node_nm_qk = 'L_' + str(layer) + '_OKt_' + str(h)
                    node_nm_vo = 'L_' + str(layer) + '_VO_' + str(h)
                    node_add_heads = 'L_' + str(layer) + '_add'
                    inner_graph.add_node(node_nm_qk, label=node_nm_qk, fillcolor='lightblue')
                    inner_graph.add_node(node_nm_vo, label=node_nm_vo, fillcolor='lightblue')
                    inner_graph.add_node(node_add_heads, label='add')
                    inner_graph.add_edge(start_node, node_nm_qk)
                    inner_graph.add_edge(node_nm_qk, node_nm_vo)
                    inner_graph.add_edge(node_nm_vo, node_add_heads)
                inner_graph.add_edge(node_add_heads, end_node)

        # Remove existing nodes between "NativeLayerNormBackward" and "AddBackward0"
        while len(to_be_removed_list) > 0:
            node = to_be_removed_list[0]
            if inner_graph.has_node(node):
                if 'AddBackward0' not in util.node_iter(inner_graph)[node]['label'].split("\n")[0]:
                    node_successors = list(inner_graph.successors(node))
                    for idx in range(len(node_successors)):
                        to_be_removed_list.append(node_successors[idx])
                        to_be_removed_list = list(set(to_be_removed_list))
                    inner_graph.remove_node(node)
                    to_be_removed_list.remove(node)
                else:
                    to_be_removed_list.remove(node)
            else:
                to_be_removed_list.remove(node)
        return inner_graph

    def do_simplify_graph(self, input_graph, num_of_head):
        input_graph = self.remove_node(input_graph)
        input_graph = self.skip_node(input_graph)
        input_graph = self.replace_node(input_graph)
        input_graph = self.add_nodes_for_each_head(input_graph, head=num_of_head)
        return input_graph

    def simplify_g(self):
        inputs = self.tokenizer("dummy input")
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v).unsqueeze(0)

        make_dot_digraph = make_dot(
            self.model(**inputs).last_hidden_state,
            params=dict(list(self.model.named_parameters())),
            show_saved=False,
            show_attrs=True)
        dotplus = pydotplus.graph_from_dot_data(make_dot_digraph.source)

        g = nx.nx_pydot.from_pydot(dotplus)
        g_simple = self.do_simplify_graph(g, self.config.num_attention_heads)

        return g_simple
