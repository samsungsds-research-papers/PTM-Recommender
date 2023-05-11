import glob, json, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_model",  default="all",   type=str,   required=False)
parser.add_argument("--source_dir", default=".", type=str, required=False)
parser.add_argument("--pretrained_model_dir", default="pretrained_models", type=str, required=False)
parser.add_argument("--data_dir", default="finetune_data/squad_v2_data", type=str, required=False)
parser.add_argument("--save_model", action="store_true", required=False)
args = parser.parse_args()
args.trainer_code=os.path.join(args.source_dir, 'finetuning_code', 'run_squad.py')
if args.save_model:
    args.save_model = 1
else:
    args.save_model = -1
model_path_list = glob.glob(args.pretrained_model_dir+'/*')   
import time
for idx, model_dir in enumerate(model_path_list):
    starting_time=time.time()
    model_name= model_dir.split('/').pop()
    if model_name.startswith(args.base_model) is False and args.base_model !='all':
        continue
    if model_name.endswith('json'):
        continue
    if os.path.isfile(model_dir+'/squad_v2/result.json') is True and args.rerun is False:
        continue
    with open(model_dir +'/config.json', 'r') as f:
        config = json.load(f)   

    batch_size = 16 if 'large' in model_name else 32
    # batch_size = 32
    command = 'python {} '.format(args.trainer_code) +\
    '--model_name_or_path {} '.format(model_dir)+ \
    '--data_dir {} '.format(args.data_dir)+ \
    '--version_2_with_negative ' + \
    '--do_train ' + \
    '--do_eval ' + \
    '--learning_rate 3e-5 ' +\
    '--num_train_epochs 2 ' + \
    '--max_seq_length 512 ' +\
    '--doc_stride 128 '+ \
    '--seed 0 ' + \
    '--overwrite_output_dir ' +\
    '--base_model {} '.format(model_name)+ \
    '--output_dir {}/squad_v2 '.format(model_dir) + \
    '--save_model {} '.format(args.save_model) + \
    '--per_gpu_train_batch_size {} '.format(batch_size)+ \
    '--model_type {}'.format(config['model_type'])
    os.system(command)
    end_time=time.time()-starting_time
    finetune_time_dir='{}/squad_v2/finetune_time.json'.format(model_dir) 
    with open(finetune_time_dir, "w") as f:
        json.dump(end_time, f)


