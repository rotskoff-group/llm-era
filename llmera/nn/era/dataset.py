import torch
from torch.utils.data import Dataset


def era_collate_fn(data):
    input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, logp_masks_1, logp_masks_2, ref_logps_y1, ref_logps_y2, energies_1, energies_2 = zip(
        *data)
    
    input_ids_y1 = torch.stack(input_ids_1)
    input_ids_y2 = torch.stack(input_ids_2)

    attention_mask_y1 = torch.stack(attention_mask_1)
    attention_mask_y2 = torch.stack(attention_mask_2)

    logp_masks_y1 = torch.stack(logp_masks_1)
    logp_masks_y2 = torch.stack(logp_masks_2)

    ref_logps_y1 = torch.stack(ref_logps_y1)
    ref_logps_y2 = torch.stack(ref_logps_y2)

    energies_y1 = torch.stack(energies_1)
    energies_y2 = torch.stack(energies_2)

    return (input_ids_y1, input_ids_y2,
            attention_mask_y1, attention_mask_y2,
            logp_masks_y1, logp_masks_y2,
            ref_logps_y1, ref_logps_y2,
            energies_y1, energies_y2)


class ERADataset(Dataset):
    def __init__(self, get_hdf5,
                 data_in_memory=True):
        self.data_in_memory = data_in_memory
        self.get_hdf5 = get_hdf5
        
        era_hdf5 = self.get_hdf5()
        self.num_examples_per_prompt = era_hdf5.attrs['num_examples_per_prompt']
        self.num_pairs_per_prompt = (self.num_examples_per_prompt 
                                     * (self.num_examples_per_prompt - 1) // 2)
        self.num_prompts = era_hdf5.attrs['num_prompts']
        assert (len(era_hdf5['input_ids'])
                == self.num_examples_per_prompt * self.num_prompts)
        del era_hdf5

        self.length = int(self.num_prompts * self.num_pairs_per_prompt)
        self.all_pairs = [[i, j] for i in range(self.num_examples_per_prompt)
                          for j in range(i + 1, self.num_examples_per_prompt)]

    def open_hdf5(self):
        self.era_hdf5 = self.get_hdf5()
        self.input_ids = self.era_hdf5['input_ids']
        self.attention_mask = self.era_hdf5['attention_mask']
        self.logp_masks = self.era_hdf5['logp_masks']
        self.logps_y = self.era_hdf5['logps_y']
        self.energies = self.era_hdf5['energies']

        if self.data_in_memory:
            self.input_ids = self.input_ids[:]
            self.attention_mask = self.attention_mask[:]
            self.logp_masks = self.logp_masks[:]
            self.logps_y = self.logps_y[:]
            self.energies = self.energies[:]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, 'era_hdf5'):
            self.open_hdf5()
        prompt_index = idx // self.num_pairs_per_prompt
        pair_index = idx % self.num_pairs_per_prompt
        pair = self.all_pairs[pair_index]
        idx_1 = prompt_index * self.num_examples_per_prompt + pair[0]
        idx_2 = prompt_index * self.num_examples_per_prompt + pair[1]
        assert idx_1 != idx_2

        return (torch.tensor(self.input_ids[idx_1]), torch.tensor(self.input_ids[idx_2]),
                torch.tensor(self.attention_mask[idx_1]), torch.tensor(self.attention_mask[idx_2]),
                torch.tensor(self.logp_masks[idx_1]), torch.tensor(self.logp_masks[idx_2]),
                torch.tensor(self.logps_y[idx_1]), torch.tensor(self.logps_y[idx_2]),
                torch.tensor(self.energies[idx_1]), torch.tensor(self.energies[idx_2]))

