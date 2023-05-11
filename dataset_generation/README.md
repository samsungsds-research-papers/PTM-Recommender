# Dataset generation process

To provide reproducibility of the dataset, we shall provide dataset generation process in detail.

## Download base models from huggingface

Download base models from https://huggingface.co/

12 base models are:

| albert-base-v2                    | albert-large-v2              | albert-large-v2                      | albert-large-v2                       |
| :-------------------------------- | :--------------------------- | :----------------------------------- | :------------------------------------ |
| **distilbert-base-uncased** | **distilroberta-base** | **electra-base-discriminator** | **electra-large-discriminator** |
| **roberta-base**            | **roberta-large**      | **xlm-roberta-base**           | **xlm-roberta-large**           |

Save 12 base_models in ***pretrained_models*** folder

## Requirements

To train gnn-based model performance predictor install

```setup
virtualenv dataset_generation_env
source dataset_generation_env/bin/activate
pip install -r requirements_data_generation.txt
```

or install packages via pip:

```
pip install datasets
pip install torch==1.12
pip install evaluate
pip install transformers
pip install accelerate
pip install -U scikit-learn
pip install tensorboard
```

To run a finetuning, we need to down load dataset for GLUE tasks and SQuAD_v2 task.

To download SQuAD_v2 glue dataset run:

```
$ mkdir finetune_data
$ mkdir finetune_data/squad_v2_data
$ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O finetune_data/squad_v2_data/train-v2.0.json
$ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O finetune_data/squad_v2_data/dev-v2.0.json
```

To download glue dataset run

```
python download_glue_dataset.py
```

To calculate LogME score, run

```
cd finetuning_code
git clone git clone https://github.com/thuml/LogME.git
```

## Finetune base models

For 12 base models, we run a finetuning code by running

```The
python train_glue.py --save_model
python train_squad.py --save_model
```

## Merge models

Once finetuning for the base models are completed we merge finetuned model by running

```
python merge_models.py
```

Then merged models are  saved in ***pretrained_models/{base_model}/merge.***

Processed dill file is saved in ***merged_models***

## Data bench generation

Once merging model is completed there are 540 PTM models. we finetune each model by

```
python train_glue.py --predtraind_model_dir pretrained_models/{base_model}/merge
python train_squad.py --predtraind_model_dir pretrained_models/{base_model}/merge
```

## LogME calculation

For GLUE tasks, we calculated LogME score by running

```
python train_glue.py --predtraind_model_dir {path_of_PTM_model_folders} --logme_only
```

## Acknowledgement

In this process, we modified publically avaible codes:

* https://github.com/huggingface/transformers/blob/main/examples/legacy/question-answering/
* ttps://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
