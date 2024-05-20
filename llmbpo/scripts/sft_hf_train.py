import hydra
import os
from omegaconf import OmegaConf
import transformers
import torch
import trl
from datasets import Dataset
import h5py
from peft import LoraConfig


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="sft_hf_train")
def main(cfg):
    model_config = cfg.model
    tokenizer_config = cfg.tokenizer
    train_config = cfg.train
    global_config = cfg.global_args

    output_dir = train_config["training_arguments_args"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, f"{output_dir}/config.yaml")

    model_class = getattr(transformers, model_config["model_name"])

    tokenizer_class = getattr(transformers,
                              tokenizer_config["tokenizer_name"])
    tokenizer = tokenizer_class.from_pretrained(
        **tokenizer_config["tokenizer_args"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_file = h5py.File(f"{global_config.train_dataset_filename}",
                           "r")
    train_tokens = {key: train_file[key][:] 
                    for key in ["input_ids", "attention_mask"]}
    train_dataset = Dataset.from_dict(train_tokens)

    val_file = h5py.File(f"{global_config.eval_dataset_filename}",
                         "r")
    val_tokens = {key: val_file[key][:]
                  for key in ["input_ids", "attention_mask"]}
    val_dataset = Dataset.from_dict(val_tokens)

    print("Training dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    training_args = transformers.TrainingArguments(
        **OmegaConf.to_object(train_config["training_arguments_args"]))

    if global_config.use_qlora:
        print("Using QLORA")
        # Hard code this for now to be consistent with HF
        # https://huggingface.co/blog/dpo-trl

        bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True,
                                                     bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=torch.bfloat16,)
        model = model_class.from_pretrained(quantization_config=bnb_config,
                                            **model_config["model_args"])
        # Hard code this for now to be consistent with HF
        # https://huggingface.co/blog/dpo-trl
        peft_config = LoraConfig(r=8,
                                 lora_alpha=16,
                                 lora_dropout=0.05,
                                 target_modules=["q_proj", "v_proj"],
                                 bias="none",
                                 task_type="CAUSAL_LM")

        trainer = trl.SFTTrainer(model, train_dataset=train_dataset,
                                 eval_dataset=val_dataset,
                                 tokenizer=tokenizer,
                                 peft_config=peft_config,
                                 args=training_args,
                                 dataset_text_field="",
                                 dataset_batch_size=1024,
                                 max_seq_length=1024)

    else:
        model = model_class.from_pretrained(torch_dtype=torch.bfloat16,
                                            **model_config["model_args"])

        trainer = trl.SFTTrainer(model, train_dataset=train_dataset,
                                 eval_dataset=val_dataset,
                                 tokenizer=tokenizer,
                                 args=training_args,
                                 dataset_text_field="",
                                 dataset_batch_size=1024,
                                 max_seq_length=1024)

    trainer.train()
