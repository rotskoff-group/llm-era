#!/bin/bash
export HF_HOME=/pscratch/sd/s/shriramc/hf_cache/
echo "HF_HOME is set to $HF_HOME"
llm_hf_sft_train "model.model_args.pretrained_model_name_or_path=meta-llama/Llama-2-13b-hf"\
                 "global_args.use_qlora=True"
