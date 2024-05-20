
import llmera.nn
import h5py
import torch
import warnings
from lightning.fabric.utilities.data import AttributeDict
import os
from omegaconf import OmegaConf
import numpy as np


h_params_era = ["model_name", "model_args", "prompt_length", "max_length",
                "beta", "gamma", "do_preference_reg", "optimizer",
                "optimizer_args", "lr_scheduler", "lr_scheduler_args", "monitor"]

h_params_sft = ["model_name", "model_args", "prompt_length", "max_length",
                "optimizer", "optimizer_args", "lr_scheduler", "lr_scheduler_args", "monitor"]


def get_hdf5_fn(dataset_filename):
    """
    Args:
        cg_dataset_filename (str): The name of the file where the dataset is saved
    Returns:
        cg_hdf5: A dictionary with the dataset
    """
    def get_hdf5_data():
        return h5py.File(dataset_filename, "r")
    return get_hdf5_data


def create_lightning_model(train_config, model_config):
    """Creates a neural network model
    Args:
        train_config (dict): A dictionary with the configuration for the training
        model_config (dict): A dictionary with the configuration for the model
    Returns:
        lightning_model: A LightningModule model
    """
    lightning_model = getattr(llmera.nn, train_config["lightning_model"])
    # Make the nn in the potential so Lightning is aware of hyperparameters
    lightning_model = lightning_model(model_config["model_name"],
                                      model_config["model_args"],
                                      **train_config["lightning_model_args"])
    return lightning_model


def create_hf_trainer(model, train_config, train_dataset,
                      eval_dataset):

    training_args_class = getattr(llmera.nn, 
                                  train_config["training_arguments"])
    training_args = training_args_class(
        **OmegaConf.to_object(train_config["training_arguments_args"]))
    
    
    data_collator = getattr(llmera.nn, train_config["collate_fn"])
    trainer_class = getattr(llmera.nn, train_config["trainer_model"])
    trainer = trainer_class(model, train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            data_collator=data_collator,
                            args=training_args)
    return trainer


def create_dataset(train_config, dataset_filename):
    dataset = getattr(llmera.nn, train_config["dataset"])
    get_hdf5 = get_hdf5_fn(dataset_filename)

    dataset = dataset(get_hdf5=get_hdf5,
                      data_in_memory=train_config["data_in_memory"])
    return dataset


def create_dataloaders(dataset, train_config):
    """Creates dataloaders from a dataset
    Args:
        dataset: A (subclassed) PyTorch Dataset
        nn_config (dict): A dictionary with the configuration for the neural network
    Returns:
        train_loader, val_loader, test_loader: PyTorch Dataloaders
    """
    train, val, test = torch.utils.data.random_split(dataset,
                                                     list(train_config["dataset_split_args"].values()))
    collate_fn = getattr(llmera.nn, train_config["collate_fn"])

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_config["batch_size"], shuffle=True,
                                               num_workers=train_config["num_workers"], collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val, batch_size=train_config["batch_size"], shuffle=False,
                                             num_workers=train_config["num_workers"], collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test, batch_size=train_config["batch_size"], shuffle=False,
                                              num_workers=train_config["num_workers"], collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def create_dataloaders(dataset, train_config):
    """Creates dataloaders from a dataset
    Args:
        dataset: A (subclassed) PyTorch Dataset
        nn_config (dict): A dictionary with the configuration for the neural network
    Returns:
        train_loader, val_loader, test_loader: PyTorch Dataloaders
    """
    train, val, test = torch.utils.data.random_split(dataset,
                                                     list(train_config["dataset_split_args"].values()))
    collate_fn = getattr(llmera.nn, train_config["collate_fn"])

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_config["batch_size"], shuffle=True,
                                               num_workers=train_config["num_workers"], collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val, batch_size=train_config["batch_size"], shuffle=False,
                                             num_workers=train_config["num_workers"], collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test, batch_size=train_config["batch_size"], shuffle=False,
                                              num_workers=train_config["num_workers"], collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def create_dataloaders_prompt(dataset, train_config):
    """Creates dataloaders from a dataset and splits prompts between different sets
    Args:
        dataset: A (subclassed) PyTorch Dataset
        nn_config (dict): A dictionary with the configuration for the neural network
    Returns:
        train_loader, val_loader, test_loader: PyTorch Dataloaders
    """
    train, val, test = torch.utils.data.random_split(range(dataset.num_prompts),
                                                     list(train_config["dataset_split_args"].values()))

    train.indices = sum([[i * dataset.num_pairs_per_prompt + j] for i in train.indices
                         for j in range(dataset.num_pairs_per_prompt)], [])
    val.indices = sum([[i * dataset.num_pairs_per_prompt + j] for i in val.indices
                       for j in range(dataset.num_pairs_per_prompt)], [])
    test.indices = sum([[i * dataset.num_pairs_per_prompt + j] for i in test.indices
                        for j in range(dataset.num_pairs_per_prompt)], [])

    train.dataset = dataset
    val.dataset = dataset
    test.dataset = dataset

    assert (len(train) + len(val) + len(test)) == len(dataset)

    collate_fn = getattr(llmera.nn, train_config["collate_fn"])

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_config["batch_size"], shuffle=True,
                                               num_workers=train_config["num_workers"], collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val, batch_size=train_config["batch_size"], shuffle=False,
                                             num_workers=train_config["num_workers"], collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test, batch_size=train_config["batch_size"], shuffle=False,
                                              num_workers=train_config["num_workers"], collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def get_all_version_matches(network_folder_name, lightning_model, task_name):
    """Gets all the versions of a network that match the current configuration
    Args:
        network_folder_name (str): The name of the folder where the network is saved
        lightning_model: A LightningModule model
    Returns:
        all_matches: A list with all the versions that match the current configuration
    """
    all_versions = np.sort([int(f.split("_")[1])
                            for f in os.listdir(network_folder_name)
                            if f.startswith("version_")
                            and os.path.isdir(network_folder_name + f)
                            and "hparams.yaml" in os.listdir(network_folder_name + f)])
    all_matches = []
    for version in all_versions:
        h_params_folder_name = f"{network_folder_name}version_{version}/"
        h_params = AttributeDict(OmegaConf.load(
            f"{h_params_folder_name}/hparams.yaml"))

        if task_name == "era":
            h_param_names_to_check = h_params_era
        elif task_name == "sft":
            h_param_names_to_check = h_params_sft
        else:
            raise ValueError("Model type not recognized")

        does_match = True

        for h_param_name in h_param_names_to_check:
            does_match = (does_match
                          and (h_params[h_param_name] == lightning_model.hparams[h_param_name]))
        if does_match:
            all_matches.append(version)
    return all_matches


def get_ckpt_path(network_folder_name, lightning_model, task_name):
    """Gets the checkpoint filename for a network
    Args:
        lightning_model: A LightningModule model
        network_folder_name (str): The name of the folder where the network is saved
    Returns:
        ckpt_filename (str): The name of the file where the checkpoint is saved
        match_num (int): The version number of the match
    """
    all_matches = get_all_version_matches(network_folder_name, lightning_model,
                                          task_name)
    match_num = None
    if len(all_matches) > 1:
        warnings.warn(
            "More than one match found for the model, using the latest version")
        match_num = all_matches[-1]
    elif len(all_matches) == 0:
        raise ValueError("No match found for the model")
    else:
        match_num = all_matches[0]
    ckpt_path = f"{network_folder_name}version_{match_num}/"
    return ckpt_path, match_num
