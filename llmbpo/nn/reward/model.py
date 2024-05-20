import lightning as L
import torch
import torch.nn as nn
import numpy as np
import transformers

# https://github.com/vwxyzjn/lm-human-preference-details/blob/ccc19538e817e98a60d325[…]a562cb49/lm_human_preference_details/train_reward_accelerate.py

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer



class RewardNN(nn.Module):
    def __init__(self, model_name, model_args):
        super().__init__()
        model_class = getattr(transformers, model_name)
        self.llm = model_class.from_pretrained(**model_args)
        self.scalar_head = layer_init(nn.Linear(self.llm.config.hidden_size, 1),
                                      std=1 / np.sqrt(self.llm.config.hidden_size + 1))
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), 
                                              requires_grad=True)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), 
                                              requires_grad=True)
        
    def forward(self, **kwargs):
        output = self.llm(output_hidden_states=True, 
                          **kwargs)
        reward_latents = output.hidden_states[-1]
        # shape: [batch_size, length, hidden_size]
        last_reward_latents = reward_latents[:, -1, :]
        # shape: [batch_size, hidden_size]
        reward = self.scalar_head(last_reward_latents)
        # shape: [batch_size, 1]
        reward = self.reward_gain * reward + self.reward_bias
        return reward

class RewardModel(L.LightningModule):
    def __init__(self, model_name, model_args,
                 prompt_length, max_length,
                 optimizer,
                 optimizer_args,
                 lr_scheduler, lr_scheduler_args,
                 monitor):
        super().__init__()
        self.model = RewardNN(model_name, model_args)
        self.model.train()
        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)
    

    def _shared_eval(self, batch, batch_idx, prefix):
        (pref_input_ids, pref_attention_mask, 
         dispref_input_ids, dispref_attention_mask) = batch
        
        num_pairs = pref_input_ids.shape[0]

        all_input_ids = torch.cat([pref_input_ids, dispref_input_ids], dim=0)
        all_attention_mask = torch.cat([pref_attention_mask, dispref_attention_mask], dim=0)

        outputs = self.model(input_ids = all_input_ids, 
                             attention_mask=all_attention_mask,
                             labels=all_input_ids)
        
        preferred_outputs = outputs[:num_pairs]
        dispreferred_outputs = outputs[num_pairs:]

        loss = -nn.functional.logsigmoid((preferred_outputs - dispreferred_outputs)).mean()
        metrics = {f"{prefix}/CELoss": loss.item()}
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)
        return loss
    

    def training_step(self, batch, batch_idx):
        assert self.model.training
        loss = self._shared_eval(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.model.training
        with torch.enable_grad():
            loss = self._shared_eval(batch, batch_idx, "val")
            
            # Computational graph doesnt clear with mixed precision
            loss.backward()
            self.model.zero_grad()
            
    def test_step(self, batch, batch_idx):
        assert not self.model.training
        with torch.enable_grad():
            loss = self._shared_eval(batch, batch_idx, "test")
            
            loss.backward()
            self.model.zero_grad()

    def configure_optimizers(self):

        u_optimizer = getattr(torch.optim,
                              self.hparams.optimizer)
        u_optimizer = u_optimizer(self.model.parameters(),
                                  **self.hparams.optimizer_args)
        u_scheduler = getattr(torch.optim.lr_scheduler,
                              self.hparams.lr_scheduler)
        u_scheduler = u_scheduler(u_optimizer,
                                  **self.hparams.lr_scheduler_args)
        to_return = {"optimizer": u_optimizer, "lr_scheduler": u_scheduler,
                     "monitor": self.hparams.monitor}
        return to_return
    
    def load_model_from_ckpt(self, filename):
        # model on original cuda:0 persists in memory for some reason
        model_weights = torch.load(filename, map_location="cpu")["state_dict"]
        model_weights = {f"llm.{k[6:]}": v for k, v in model_weights.items()
                         if k.startswith("model.")}
        self.model.load_state_dict(model_weights, strict=False)
