import hydra
import os
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from llmera.nn import (create_lightning_model,
                       create_dataset, create_dataloaders,
                       get_ckpt_path)


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="reward_train")
def main(cfg):
    model_config = cfg.model
    train_config = cfg.train
    global_config = cfg.global_args

    L.seed_everything(**train_config["seed_args"])

    bpo_model = create_lightning_model(train_config=train_config,
                                       model_config=model_config)

    dataset = create_dataset(train_config=train_config,
                             dataset_filename=f"{global_config.root_data_folder_name}{global_config.task_name}/reward/reward_{train_config.lightning_model_args.prompt_length}_{train_config.lightning_model_args.max_length}.hdf5")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(dataset,
                                                                           train_config)

    resume_training_path = train_config["resume_training_path"]
    if resume_training_path is not None:
        ckpt_path, version_num = get_ckpt_path(resume_training_path,
                                               lightning_model=bpo_model)

        best_checkpoint_callback = ModelCheckpoint(filename="best_model",
                                                   monitor=train_config["lightning_model_args"]["monitor"],
                                                   mode="min",
                                                   save_top_k=1,
                                                   dirpath=f"{ckpt_path}/checkpoints")
        logger = TensorBoardLogger(save_dir="./",
                                   version=version_num)

        trainer = L.Trainer(callbacks=[best_checkpoint_callback],
                            logger=logger,
                            **train_config["trainer_args"])

        trainer.fit(bpo_model, train_dataloader, val_dataloader,
                    ckpt_path=f"{ckpt_path}/checkpoints/best_model.ckpt")

    else:
        best_checkpoint_callback = ModelCheckpoint(filename="best_model",
                                                   monitor=train_config["lightning_model_args"]["monitor"],
                                                   mode="min",
                                                   save_top_k=1)
        trainer = L.Trainer(callbacks=[best_checkpoint_callback],
                            **train_config["trainer_args"])
        if trainer.global_rank == 0:
            train_folder_name = trainer.logger.log_dir

            os.makedirs(train_folder_name, exist_ok=True)
            OmegaConf.save(cfg, f"{train_folder_name}/config.yaml")

        if train_config["load_model_filename"] is not None:
            bpo_model.load_model_from_ckpt(train_config["load_model_filename"])

        trainer.fit(bpo_model, train_dataloader, val_dataloader)
