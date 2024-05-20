import hydra
import os
from omegaconf import OmegaConf
import transformers
import torch
from llmbpo.nn import (create_hf_trainer,
                       create_dataset)

@hydra.main(version_base="1.3", config_path="../cfgs", config_name="bpo_hf_train")
def main(cfg):
    model_config = cfg.model
    train_config = cfg.train
    global_config = cfg.global_args



    output_dir = train_config["training_arguments_args"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, f"{output_dir}/config.yaml")

    model_class = getattr(transformers, model_config["model_name"])

    model = model_class.from_pretrained(torch_dtype=torch.bfloat16,
                                        **model_config["model_args"])
    train_dataset = create_dataset(train_config=train_config,
                                   dataset_filename=global_config["train_dataset_filename"])
    eval_dataset = create_dataset(train_config=train_config,
                                  dataset_filename=global_config["eval_dataset_filename"])

    trainer = create_hf_trainer(model, train_config, 
                                train_dataset, eval_dataset)
    
    trainer.train()
