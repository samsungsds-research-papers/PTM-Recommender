from transformers import AutoModel, AutoConfig, AutoTokenizer
import dill
import sys
sys.path.append('..')
from ptm_recommender.merge_finetuned_models import merge_models as mm
from ptm_recommender.graph_generator import graph_generator
from ptm_recommender.dict_nn_graph import dict_nn_graph
from ptm_recommender.dict_nn_graph_for_albert import dict_nn_graph_for_albert
import os
import torch


torch.set_num_threads(2)
PTM_PREFIX = "pretrained_models/"
FTM_PREFIX = PTM_PREFIX 
RATIO = [0.5]
GRAPH_SAVE_DIR = "merged_models/"

MODELS = [
    "albert-base-v2",   
    "albert-large-v2",     
    "bert-base-uncased",    
    "bert-large-uncased",   
    "distilbert-base-uncased", 
    "distilroberta-base",   
    "electra-base-discriminator",  
    "electra-large-discriminator",  
    "roberta-base",     
    "roberta-large",    
    "xlm-roberta-base",    
    "xlm-roberta-large"  
]

TASKS = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
    "squad_v2"
]

def get_ftm_path(ftm_dir, task):
    if task == "squad_v2":
        ftm_path = ftm_dir + "/squad_v2/finetune_model_squad_v2/"
    else:
        ftm_path = ftm_dir + "/glue/" + task + "/finetuned_model_" + task + "/"
    return ftm_path


def get_graph_info(ptm_nm, merged_model_path):
    graphs_info = []
    config = AutoConfig.from_pretrained(merged_model_path)
    model = AutoModel.from_pretrained(merged_model_path)

    m = config.num_hidden_layers
    d = config.hidden_size
    h = config.num_attention_heads

    g_func = graph_generator(num_of_enc_layer=m,
                             num_of_enc_head=h,
                             num_of_dec_layer=0,
                             num_of_dec_head=0,
                             is_encoder=True,
                             is_self=True)
    G_graph = g_func.gen_graph()

    if ptm_nm.startswith('albert'):
        dict_func = dict_nn_graph_for_albert(model, config)
    else:
        dict_func = dict_nn_graph(model, config, is_encoder=True)
    G_dict = dict_func.get_model_dict()

    fullname = ptm_nm + "_" + merged_model_path.split("/")[-2]
    graphs_info.append((G_graph, G_dict, fullname))
    print("Generated graph info of " + fullname)

    saved_path = GRAPH_SAVE_DIR + "merged_" + ptm_nm + "/"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    with open(saved_path + fullname + '.dill', 'wb') as f:
        dill.dump(graphs_info, f)


for model in MODELS:
    ptm = AutoModel.from_pretrained(PTM_PREFIX + model)
    ptm_config = AutoConfig.from_pretrained(PTM_PREFIX + model)
    ptm_tokenizer = AutoTokenizer.from_pretrained(PTM_PREFIX + model, use_fast=False)
    for task1, task2 in zip(TASKS, TASKS[1:]):
        # load fine-tuned models
        print(get_ftm_path(FTM_PREFIX + model, task1))
        ftm_1 = AutoModel.from_pretrained(get_ftm_path(FTM_PREFIX + model, task1))
        ftm_2 = AutoModel.from_pretrained(get_ftm_path(FTM_PREFIX + model, task2))

        for ratio in RATIO:
            # save merged as:
            # {ptm_nm}_{task_1}_{task_2}_{ratio}
            print("FTM_PREFIX : ", FTM_PREFIX)
            print("model : ", model)
            out_dir = FTM_PREFIX + model + "/merge/" + task1 + "_" + task2 + "_" + str(ratio) + "/"
            # merge weights
            merge_model = mm(ptm, ptm_config, ptm_tokenizer, ftm_1, ftm_2, ratio, out_dir)

            # generate graph info
            get_graph_info(model, out_dir)


