#!/bin/bash
export HF_HOME=/pscratch/sd/s/shriramc/hf_cache/
echo "HF_HOME is set to $HF_HOME"
llm_bpo_train "train=bpo"\
    "global_args.root_data_folder_name=/pscratch/sd/s/shriramc/llm_datasets/"\
    "global_args.task_name=llama"\
    "model=llama"\
    "train.batch_size=1"\
    "train.trainer_args.precision=bf16-true"\
    "train.load_model_filename=null"\
    "train.lightning_model_args.beta=1.0"\
    "train.lightning_model_args.gamma=0.1"\
    "train.lightning_model_args.prompt_length=-1"\
    "train.lightning_model_args.max_length=-1"\
    "train.trainer_args.devices=4"\