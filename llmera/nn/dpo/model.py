import lightning as L
import torch
import transformers
import torch
import torch.nn as nn


class DPOModel(L.LightningModule):
    def __init__(self, model_name, model_args,
                 prompt_length, max_length,
                 beta,
                 do_preference_reg, optimizer,
                 optimizer_args,
                 lr_scheduler, lr_scheduler_args,
                 monitor):
        super().__init__()
        model_class = getattr(transformers, model_name)
        self.model = model_class.from_pretrained(**model_args)
        self.model.train()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        """Used for computing log probabilities
        """
        assert not self.model.training
        input_ids, attention_mask, logp_masks, sample_idx = batch

        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask}

        policy_logits = self.model(**inputs)['logits']
        policy_per_token_logps = torch.gather(policy_logits[:, :-1, :].log_softmax(-1),
                                              dim=2, index=inputs["input_ids"][:, 1:].unsqueeze(2)).squeeze(2)

        # Apply logp_masks (mask over padding and prompt tokens)
        policy_per_token_logps = (policy_per_token_logps
                                  * logp_masks[:, 1:])
        policy_logps_y = policy_per_token_logps.sum(-1)

        return policy_logps_y, sample_idx

    def _shared_eval(self, batch, batch_idx, prefix):
        (input_ids_y1, input_ids_y2,
         attention_mask_y1, attention_mask_y2,
         logp_masks_y1, logp_masks_y2,
         ref_logps_y1, ref_logps_y2,
         energies_y1, energies_y2) = batch
        
        num_pairs = input_ids_y1.shape[0]
        assert (num_pairs == input_ids_y2.shape[0] 
                == attention_mask_y1.shape[0] == attention_mask_y2.shape[0] 
                == logp_masks_y1.shape[0] == logp_masks_y2.shape[0] 
                == ref_logps_y1.shape[0] == ref_logps_y2.shape[0] 
                == energies_y1.shape[0] == energies_y1.shape[0])


        inputs = {"input_ids": torch.cat([input_ids_y1, input_ids_y2], dim=0),
                  "attention_mask": torch.cat([attention_mask_y1, attention_mask_y2], dim=0)}
        
        logp_masks = torch.cat([logp_masks_y1, logp_masks_y2], dim=0)

        policy_logits = self.model(**inputs)['logits']
        policy_per_token_logps = torch.gather(policy_logits[:, :-1, :].log_softmax(-1),
                                              dim=2, index=inputs["input_ids"][:, 1:].unsqueeze(2)).squeeze(2)
        
        # Apply logp_masks (mask over padding and prompt tokens)
        policy_per_token_logps = (policy_per_token_logps
                                  * logp_masks[:, 1:])
        policy_logps_y = policy_per_token_logps.sum(-1)

        policy_logps_y1 = policy_logps_y[:num_pairs]
        policy_logps_y2 = policy_logps_y[num_pairs:]

        assert policy_logps_y1.shape == policy_logps_y2.shape

        y2_sign = (energies_y2 >= energies_y1).long()
        y2_sign[y2_sign == 0] = -1
        y1_sign = -y2_sign

        pi_logratios = (y2_sign * policy_logps_y2 
                        + y1_sign * policy_logps_y1)
        ref_logratios = (y2_sign * ref_logps_y2
                        + y1_sign * ref_logps_y1)
        loss = nn.functional.logsigmoid(self.hparams.beta * (pi_logratios - ref_logratios))
        loss = loss.mean()
 

        metrics = {f"{prefix}/loss": loss.item()}
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
        model_weights = {k[6:]: v for k, v in model_weights.items()
                         if k.startswith("model.")}
        self.model.load_state_dict(model_weights)
