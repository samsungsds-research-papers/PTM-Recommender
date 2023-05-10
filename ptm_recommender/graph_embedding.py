import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import dill
import os
import ptm_recommender.graph_models.graph_util as util
import json, glob
from scipy import stats


def load_task_performance(args, task_names):    
    performance_dir=os.path.join(args.datadir, 'performance', 'merged_models')
    task_performance={}
    for task in task_names:
        json_file_path = os.path.join(performance_dir, task+'_performance_score.json')
        with open(json_file_path, 'r') as f:
            task_performance[task]= json.load(f)[task]  
    return task_performance

class graph_embedding:
    def __init__(self, args, task_lists, feature_dim=128):
        self.args = args
        self.task_results  = load_task_performance(args, task_lists)
        self.feature_dim = feature_dim
        self.load_graphs()
        self.task_to_metric = {
            "mrpc" : "eval_accuracy",
            "cola" : "eval_matthews_correlation",
            "rte" :'eval_accuracy',
            "sst2" :'eval_accuracy',
            "stsb" :'eval_pearson',
            "wnli" :'eval_accuracy',
            "qnli" :"eval_accuracy",
            "mnli" : "eval_accuracy",
            "qqp"  : "eval_accuracy",
            "squad_v2":'f1',
        }
        
        if os.path.isdir(self.args.results_dir) is False:
            os.makedirs(self.args.results_dir)
        self.normalizer = None
        self.original_acciracy=[]
                  
        
        
    def load_graphs(self):
        graph_info=[]
        graph_file_dir =os.path.join(self.args.datadir, self.args.node_feat_type)
        graph_file_list = glob.glob(graph_file_dir+'/*')
        for each_file_path in graph_file_list:
            if each_file_path .endswith('.dill') is False:
                continue
            with open(each_file_path, 'rb') as f:
                graph_obj = dill.load(f)
            graph_info += graph_obj
        self.graph_info_graphs = [x[0] for x in graph_info]
        self.graph_info_params = [x[1] for x in graph_info]
        self.graph_info_model_nm = [x[2] for x in graph_info]
        self.get_node_feature()
        self.train_dataset = None
        self.test_dataset =None     
            
    def get_node_feature(self):
        for (G, p) in zip(self.graph_info_graphs, self.graph_info_params):
            G.graph['feat_dim'] = self.feature_dim
            param_keys = p.keys()
            for u in util.node_iter(G):
                label = util.node_dict(G)[u]['label']
                if label in param_keys:
                    feature = p[label]
                    if feature.shape[0] != self.feature_dim:
                        assert feature.shape[0] == self.feature_dim
                else:
                    feature = torch.zeros(self.feature_dim).squeeze()
                util.node_dict(G)[u]['feat'] = feature.float()
        
    def get_graph_acc_model_nms(self, task):        
        model_nm_to_accuracies = self.task_results[task]
        graphs, params, model_nm, accuracies = [], [], [], []
        for idx, full_name in enumerate(self.graph_info_model_nm):
            if full_name in model_nm_to_accuracies:
                graphs.append(self.graph_info_graphs[idx])
                params.append(self.graph_info_params[idx])
                model_nm.append(self.graph_info_model_nm[idx])
                if task=='squad_v2':
                    accuracies.append(model_nm_to_accuracies[full_name][self.task_to_metric[task]]/100)
                else:
                    accuracies.append(model_nm_to_accuracies[full_name][self.task_to_metric[task]])   
        return graphs, accuracies, model_nm
    
    def generate_train_test_dataset(self, task, model_type, seed=0):
        from ptm_recommender.graph_models.gcnn.gin_utils import graph_to_s2vgraphs
        
        graphs, accuracies, model_nms = self.get_graph_acc_model_nms(task)
        graphs_accuracies_nms = [(graphs[i], accuracies[i], model_nms[i]) for i in range(len(graphs))]               

        random.seed(seed)
        random.shuffle(graphs_accuracies_nms)
        idx_train = int(self.args.train_ratio * len(graphs_accuracies_nms))
        train_ga, test_ga = graphs_accuracies_nms[:idx_train], graphs_accuracies_nms[idx_train:]

        train_graphs = [x[0] for x in train_ga]
        train_accuracies = [x[1] for x in train_ga]
        train_model_nms = [x[2] for x in train_ga]
        
        test_graphs = [x[0] for x in test_ga]
        test_accuracies = [x[1] for x in test_ga]
        test_model_nms = [x[2] for x in test_ga]
        

        train_dataset = graph_to_s2vgraphs(train_graphs, train_accuracies, train_model_nms)
        test_dataset = graph_to_s2vgraphs(test_graphs, test_accuracies, test_model_nms)         
        return train_dataset, test_dataset, None
        
    
    def load_test_accuracy(self, task):
        model_nm_to_accuracies =  self.task_results[task]
        for idx, data in enumerate(self.train_dataset):
            if task=='squad_v2':
                data.accuracy = model_nm_to_accuracies[data.model_nm][self.task_to_metric[task]]/100
            else:
                data.accuracy = model_nm_to_accuracies[data.model_nm][self.task_to_metric[task]]
        for idx, data in enumerate(self.test_dataset):
            if task=='squad_v2':
                data.accuracy = model_nm_to_accuracies[data.model_nm][self.task_to_metric[task]]/100
            else:
                data.accuracy = model_nm_to_accuracies[data.model_nm][self.task_to_metric[task]]

     
    def run_gnn(self, task, seed = 0 ):
        import time
        starting_time = time.time()
        criterion = nn.L1Loss()
        self.train_dataset, self.test_dataset, self.dataset = \
            self.generate_train_test_dataset(task, model_type=self.args.model_type, seed=seed)
        test_dataset, model, train_loss, test_loss, ypreds = \
            self.run_gcnn(task, self.train_dataset, self.test_dataset, criterion)
        test_acc = [x.accuracy for x in test_dataset]
        test_model_nm = [x.model_nm for x in test_dataset]
        
        ypreds = np.array([x.item() for x in ypreds])
        ypreds_argmax = np.argmax(ypreds)
        max_ypred = ypreds[ypreds_argmax]
        
        true_max_ypred = test_acc[ypreds_argmax]
        selected_model_name = test_model_nm[ypreds_argmax]
        train_loss = train_loss.item()
        test_loss = test_loss.item()
    
        
        total_result = [{'true_accuracy': acc,
                         'pred_accuracy': pred,
                         'model_name': model_name}
                        for acc, pred, model_name in zip(test_acc, list(ypreds), test_model_nm)]
        training_time = time.time()-starting_time
        random_acc_idx = random.choice(range(len(test_acc)))
        random_acc = test_acc[random_acc_idx]
        random_model_name = test_model_nm[random_acc_idx]
        fold_result = {'selection': {'model': {'true_accuracy': true_max_ypred,
                                               'pred_accuracy': max_ypred,
                                               'model_name': selected_model_name},
                                     'random': {'true_accuracy': random_acc,
                                                'pred_accuracy': ypreds[random_acc_idx],
                                                'model_name': random_model_name},
                                     },
                       'model_result': {'train_mae': train_loss,
                                        'test_mae': test_loss},
                       'total_result': total_result,
                       'args': self.args,
                       'task': task,
                       'training_time': training_time
                       }
        if self.args.output_name is None:
            result_path = os.path.join(self.args.results_dir, 'result_{}.dill'.format(task))
        else:
            result_path = os.path.join(
                self.args.results_dir,
                '{}_{}.dill'.format(self.args.output_name, task))
        with open(result_path, 'wb') as f:
            dill.dump(fold_result, f)
                
        if self.args.model_save:
            if self.args.output_name is None:
                model_path = os.path.join(self.args.results_dir, 'model_result_{}.pth'.format(task))
            else:
                model_path = os.path.join(
                    self.args.results_dir,
                    'model_{}_{}.pth'.format(self.args.output_name, task))
            
            torch.save(model, model_path)
        return fold_result

    def run_gcnn(self, task, train_dataset, test_dataset, criterion):
        from ptm_recommender.graph_models.gcnn.graphcnn import GraphCNN
        from ptm_recommender.graph_models.gcnn.gin_utils import gcnn_train, gcnn_evaluate, gcnn_save_feature_vector
        import torch.optim as optim
        
        model = GraphCNN(
                num_layers=self.args.num_gc_layers,
                num_mlp_layers=3,
                input_dim=self.feature_dim,
                hidden_dim=self.args.hidden_dim,
                output_dim=self.args.output_dim,
                final_dropout=self.args.dropout,
                learn_eps=True,
                graph_pooling_type=self.args.pooling_type,
                neighbor_pooling_type=self.args.pooling_type,
                device=self.args.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if  hasattr(self.args, 'seed'):
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)       

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        model.to(self.args.device)

        for epoch in tqdm(range(1, self.args.num_epochs + 1)):
            scheduler.step()
            train_loss, model = gcnn_train(self.args, model, train_dataset, optimizer, epoch, criterion)

        test_loss, ypreds = gcnn_evaluate(self.args, model, test_dataset)
        print("End of training")

        if self.args.save_feature:
            gcnn_save_feature_vector(self.args, model, test_dataset, task)
                
        print("args : " , self.args)
        print("*"*100)
        
        return test_dataset, model, train_loss, test_loss, ypreds
