from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
from collections import defaultdict


class DPOTrainingArguments(TrainingArguments):
    def __init__(self, beta, do_preference_reg, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.do_preference_reg = do_preference_reg

class DPOTrainer(Trainer):
    def __init__(self, model, train_dataset, 
                 eval_dataset, data_collator, args):
        super().__init__(model=model, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, data_collator=data_collator,
                         args=args)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))


    def era_loss(self, model, inputs):
        (input_ids_y1, input_ids_y2,
         attention_mask_y1, attention_mask_y2,
         logp_masks_y1, logp_masks_y2,
         ref_logps_y1, ref_logps_y2,
         energies_y1, energies_y2) = inputs["data"]

        num_pairs = input_ids_y1.shape[0]
        assert (num_pairs == input_ids_y2.shape[0]
                == attention_mask_y1.shape[0] == attention_mask_y2.shape[0]
                == logp_masks_y1.shape[0] == logp_masks_y2.shape[0]
                == ref_logps_y1.shape[0] == ref_logps_y2.shape[0]
                == energies_y1.shape[0] == energies_y1.shape[0])

        inputs = {"input_ids": torch.cat([input_ids_y1, input_ids_y2], dim=0),
                  "attention_mask": torch.cat([attention_mask_y1, attention_mask_y2], dim=0)}

        logp_masks = torch.cat([logp_masks_y1, logp_masks_y2], dim=0)

        policy_logits = model(**inputs)['logits']
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
        loss = nn.functional.logsigmoid(self.args.beta * (pi_logratios - ref_logratios))
        loss = loss.mean()

        return loss
    
    def get_batch_loss_metrics(self, model, inputs, train_eval):
        loss = self.era_loss(model, inputs)
        
        
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}dpo_loss"] = loss.item()
        return loss, metrics


    def compute_loss(self, model, inputs, return_outputs=False):

        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics, train_eval):
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs):
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
    
    def prediction_step(self, model, inputs, prediction_loss_only, 
                        ignore_keys=None):
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)
        else:
            return (loss, metrics)
