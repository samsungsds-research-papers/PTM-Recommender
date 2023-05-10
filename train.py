import sys
from ptm_recommender.graph_embedding import graph_embedding
import argparse
import os


# Setup argument
parser = argparse.ArgumentParser()
parser.add_argument("--train_ratio", default=0.8, type=float, required=False)
parser.add_argument("--dim", default=128, type=int, required=False)
parser.add_argument("--model_type", default="gcnn", type=str, required=False)
parser.add_argument("--num_gc_layers", default=5, type=int, required=False)
parser.add_argument("--hidden_dim", default=600, type=int, required=False)
parser.add_argument("--output_dim", default=1, type=int, required=False)
parser.add_argument("--dropout", default=0.4, type=float, required=False)
parser.add_argument("--pooling_type", default="sum", type=str, required=False)
parser.add_argument('--device', type=int, default=0, required=False)
parser.add_argument("--lr", default=0.001, type=float, required=False)
parser.add_argument("--num_epochs", default=200, type=int, required=False)
parser.add_argument("--iters_per_epoch", default=200, type=int,required=False)
parser.add_argument("--batch_size", default=8, type=int, required=False)
parser.add_argument("--node_feat_type", default='merged_model_dataset', type=str, required=False)
parser.add_argument("--datadir", default='./ptm_model_bench/', type=str, required=False)
parser.add_argument("--results_dir", default='./results', type=str, required=False)
parser.add_argument("--output_name", default=None, type=str, required=False)
parser.add_argument("--save_feature", action="store_true",  required=False, default=True)
parser.add_argument("--model_save", action="store_true",  required=False, default=True)
parser.add_argument('--seed', default=0, type=int, required=False)

args = parser.parse_args()
if os.path.isdir(args.results_dir) is False:
    os.mkdir(args.results_dir)
if args.node_feat_type == 'merged_model_dataset':
    args.results_dir = os.path.join(args.results_dir, args.model_type + '_ptm_model_bench')    
else:
    args.results_dir = os.path.join(args.results_dir, args.model_type + '_' + args.node_feat_type)
if os.path.isdir(args.results_dir) is False:
    os.mkdir(args.results_dir)
print("args  " , args)
task_lists = ["squad_v2", "cola", "mrpc", "rte", "sst2", "stsb", "mnli", "qnli", "qqp"]
gnn_func=graph_embedding(args,  task_lists=task_lists, feature_dim=args.dim)
for idx, task in enumerate(task_lists):
    print("Task : " , task)
    gnn_func.run_gnn(task, args.seed)