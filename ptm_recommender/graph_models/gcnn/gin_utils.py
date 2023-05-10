import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm


class S2VGraph(object):
    def __init__(self, g, accuracy, model_nm, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.accuracy = accuracy
        self.model_nm = model_nm
        self.g = g
        origin_to_int_dict = {}
        for idx, node in enumerate(g):
            origin_to_int_dict[node] = idx
        self.node_tags = list(range(len(g)))
        self.max_neighbor = 0
        neighbors = []
        for node in g.nodes():
            each_neighbors = []
            for i in g.successors(node):
                each_neighbors.append(origin_to_int_dict[i])
            for i in g.predecessors(node):
                each_neighbors.append(origin_to_int_dict[i])
            neighbors.append(each_neighbors)
            self.max_neighbor = max(self.max_neighbor, len(each_neighbors))
        self.neighbors = neighbors
        self.node_features = []
        for node in g.nodes():
            self.node_features.append(g.nodes()[node]['feat'].squeeze())
        self.node_features = torch.stack(self.node_features)
        self.edge_mat = torch.LongTensor([[origin_to_int_dict[edge[0]] for edge in g.edges()],
                                          [origin_to_int_dict[edge[1]] for edge in g.edges()]])


def graph_to_s2vgraphs(graphs, accuracies, model_nms):
    s2vgraphs = []
    for graph, accuracy, model_nm in zip(graphs, accuracies, model_nms):
        s2v_g = S2VGraph(graph, accuracy, model_nm)
        s2vgraphs.append(s2v_g)
    return s2vgraphs


def seperate_s2vgraphs(s2vgraphs, args, val_idx):
    max_num_nodes = max([G.g.number_of_nodes() for G in s2vgraphs])
    input_dim = args.dim

    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    n = len(s2vgraphs) // 10
    graph_lists = list(divide_chunks(s2vgraphs, n))
    val_dataset = graph_lists[(val_idx + 1) % 10]
    train_dataset = []
    for i in range(10):
        if i != (val_idx + 1) % 10:
            train_dataset += graph_lists[i]
    return train_dataset, val_dataset, max_num_nodes, input_dim


def gcnn_evaluate(args, model, test_graphs):
    model.eval()

    accuracy = torch.FloatTensor(
        [x.accuracy for x in test_graphs]).to(args.device)
    ypreds =[]
    mae_loss = nn.L1Loss()
    loss = 0
    for idx, test_graph in enumerate(test_graphs):
        ypred = model([test_graph])
        # print("ypred : ", ypred)
        ypred = ypred.cpu().detach()
        ypreds.append(ypred)
        loss += mae_loss(ypred, accuracy[idx].flatten().cpu())     
    # 
    print("test mae loss: %f" % (loss))
    loss = loss / len(test_graphs)
    return loss, ypreds

def customized_loss(output, target):
    mae_loss = nn.L1Loss(reduction='none')
    cross_entropy_loss = (-1)* torch.sum(torch.log(output +1e-7)*target) /len(target)
    regress_loss = (((10. * target)**2 +(10. * (1-target))**2 + 1)* mae_loss(output, target)).mean()
    return regress_loss + 10*cross_entropy_loss 


def gcnn_train(args, model, train_graphs, optimizer, epoch, criterion):
    model.train()
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    mae_loss = nn.L1Loss()
    for _ in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[
            :args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)
        target_accuracies = torch.FloatTensor([graph.accuracy for graph in batch_graph]).reshape(len(batch_graph),
                                                                                                 1).to(args.device)
        # compute loss
        loss = 100*mae_loss(output, target_accuracies).mean()
        # ((10. * target_accuracies)**2 *      
        # loss = customized_loss(output, target_accuracies)
        
        '''
        currently we are experiment with different loss functions
        mse_loss = nn.MSELoss() is most commone loss
        '''
        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))
    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss, model


def gcnn_save_feature_vector(args, model, test_graphs, task):
    import os
    import dill
    model.eval()
    print("test graphs : ", len(test_graphs))

    after_dropout, after_linear_layer, pooled_h = [], [], []
    for idx, test_graph in enumerate(test_graphs):
        # print("idx :", idx)
        v1, v2, v3 = model.get_feature([test_graph])
        after_dropout.append(v1.cpu().detach().numpy())
        after_linear_layer.append(v2.cpu().detach().numpy())
        pooled_h.append(v3.cpu().detach().numpy())

    after_dropout = np.stack(after_dropout, axis=0)
    after_linear_layer = np.stack(after_linear_layer, axis=0)
    pooled_h = np.stack(pooled_h, axis=0)
    if args.output_name is None:
        feat_vect_path = os.path.join(
            args.results_dir, 'feat_vec_result_{}.dill'.format(task))
    else:
        feat_vect_path = os.path.join(
            args.results_dir, 'feat_vec_{}_{}.dill'.format(args.output_name, task))
    accuracy = torch.FloatTensor([x.accuracy for x in test_graphs]).to('cpu')    
    feat_vects = {'after_dropout':     after_dropout,
                  'after_linear_layer': after_linear_layer,
                  'pooled_h': pooled_h,
                  'acc': accuracy, }
    with open(feat_vect_path, 'wb') as f:
        dill.dump(feat_vects, f)
     