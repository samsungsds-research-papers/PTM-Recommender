from datasets import load_dataset
for task in ['cola', 'mrpc', 'stsb', 'sst2', 'mnli', 'qnli', 'wnli', 'rte', 'qqp']:
    dataset = load_dataset('glue', task)
    dataset.save_to_disk(f'finetune_data/glue_data/{task}')