import glob, json, os, argparse,shutil
parser = argparse.ArgumentParser()
parser.add_argument("--base_model",  default="all",   type=str,   required=False)
parser.add_argument("--source_dir", default='.', type=str, required=False)
parser.add_argument("--pretrained_model_dir", default="pretrained_models", type=str, required=False)
parser.add_argument("--data_dir", default="finetune_data/glue_data", type=str, required=False)
parser.add_argument("--task_names", nargs="*",
                    default=[ "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb",  "wnli"], type=str, required=False)
parser.add_argument("--logme_only", action="store_true", required=False)
parser.add_argument("--save_model", action="store_true", required=False)

args = parser.parse_args()
args.trainer_code=os.path.join(args.source_dir,  'finetuning_code' , 'run_glue_no_trainer.py')
if args.save_model:
    args.save_model = 1
else:
    args.save_model = -1
import time
model_path_list = glob.glob(args.pretrained_model_dir+'/*')  
print(model_path_list)
for idx, model_dir in enumerate(model_path_list):
    model_name=model_dir.split('/').pop()
    if model_name.endswith('json'):
        continue
    if model_name.startswith(args.base_model) is False and args.base_model!='all':
        continue
    for TASK_NAME in args.task_names:  
        start_time=time.time()
        if args.logme_only:
            if os.path.isfile(model_dir+'/glue/{}/F_y_score.dill'.format(TASK_NAME)) is True:
                continue
            command ='python {} '.format(args.trainer_code) +  \
                '--model_name_or_path {} '.format(model_dir) + \
                '--task_name {} '.format(TASK_NAME) + \
                '--max_length 512 '+ \
                '--per_device_train_batch_size 16 ' +\
                '--learning_rate 2e-5 ' + \
                '--num_train_epochs 3 ' + \
                '--logme_only ' +\
                '--seed 0 ' + \
                '--data_dir {} '.format(args.data_dir+'/'+TASK_NAME) +\
                '--source_dir {} '.format(args.source_dir) +\
                '--output_dir {}/glue/{}/ '.format(model_dir, TASK_NAME) 
            os.system(command) 
            end_time=time.time() - start_time
            finetune_time_dir='{}/glue/{}/logme_time.json'.format(model_dir, TASK_NAME) 
            with open(finetune_time_dir, "w") as f:
                json.dump(end_time, f)
        else:
            if os.path.isfile(model_dir+'/glue/{}/all_results.json'.format(TASK_NAME)) is True:
                continue
            lr = 5e-6 if 'large' in model_dir else 2e-5
            batch_size = 16 if 'large' in model_dir else 32

            command ='python {} '.format(args.trainer_code) +  \
                '--model_name_or_path {} '.format(model_dir) + \
                '--task_name {} '.format(TASK_NAME) + \
                '--max_length 512 '+ \
                '--per_device_train_batch_size {} '.format(batch_size) +\
                '--learning_rate {} '.format(lr)   + \
                '--num_train_epochs 3 ' + \
                '--data_dir {} '.format(args.data_dir+'/'+TASK_NAME) +\
                '--seed 0 ' +\
                '--save_model {} '.format(args.save_model) + \
                '--ignore_mismatched_sizes ' +\
                '--source_dir {} '.format(args.source_dir) +\
                '--output_dir {}/glue/{}/ '.format(model_dir, TASK_NAME) 
            os.system(command)       
            end_time=time.time() -start_time
            finetune_time_dir='{}/glue/{}/finetune_time.json'.format(model_dir, TASK_NAME) 
            with open(finetune_time_dir, "w") as f:
                json.dump(end_time, f)

            
        
    