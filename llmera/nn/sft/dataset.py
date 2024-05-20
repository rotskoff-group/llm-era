import torch
from torch.utils.data import Dataset


def sft_collate_fn(data):
    input_ids, attention_mask = zip(*data)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    return input_ids, attention_mask


class SFTDataset(Dataset):
    def __init__(self, get_hdf5,
                 data_in_memory=True):
        self.data_in_memory = data_in_memory
        self.get_hdf5 = get_hdf5
        sft_hdf5 = self.get_hdf5()
        self.length = len(sft_hdf5['input_ids'])
        del sft_hdf5

    def open_hdf5(self):
        self.sft_hdf5 = self.get_hdf5()
        self.input_ids = self.sft_hdf5['input_ids']
        self.attention_mask = self.sft_hdf5['attention_mask']

        if self.data_in_memory:
            self.input_ids = self.input_ids[:]
            self.attention_mask = self.attention_mask[:]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, 'sft_hdf5'):
            self.open_hdf5()
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.attention_mask[idx])

