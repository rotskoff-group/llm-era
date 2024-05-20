import torch
from torch.utils.data import Dataset


def reward_collate_fn(data):
    (pref_input_ids, pref_attention_mask,
     dispref_input_ids, dispref_attention_mask) = zip(*data)
    pref_input_ids = torch.stack(pref_input_ids)
    pref_attention_mask = torch.stack(pref_attention_mask)
    dispref_input_ids = torch.stack(dispref_input_ids)
    dispref_attention_mask = torch.stack(dispref_attention_mask)
    return pref_input_ids, pref_attention_mask, dispref_input_ids, dispref_attention_mask


class RewardDataset(Dataset):
    def __init__(self, get_hdf5,
                 data_in_memory=True):
        self.data_in_memory = data_in_memory
        self.get_hdf5 = get_hdf5
        reward_hdf5 = self.get_hdf5()
        self.length = len(reward_hdf5['pref_input_ids'])
        del reward_hdf5

    def open_hdf5(self):
        self.reward_hdf5 = self.get_hdf5()
        self.pref_input_ids = self.reward_hdf5['pref_input_ids']
        self.pref_attention_mask = self.reward_hdf5['pref_attention_mask']

        self.dispref_input_ids = self.reward_hdf5['dispref_input_ids']
        self.dispref_attention_mask = self.reward_hdf5['dispref_attention_mask']

        if self.data_in_memory:
            self.pref_input_ids = self.pref_input_ids[:]
            self.pref_attention_mask = self.pref_attention_mask[:]

            self.dispref_input_ids = self.dispref_input_ids[:]
            self.dispref_attention_mask = self.dispref_attention_mask[:]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, 'reward_hdf5'):
            self.open_hdf5()
        return (torch.tensor(self.pref_input_ids[idx]),
                torch.tensor(self.pref_attention_mask[idx]),
                torch.tensor(self.dispref_input_ids[idx]),
                torch.tensor(self.dispref_attention_mask[idx]))
