import lightning as L
import torch
import transformers


class SFTModel(L.LightningModule):
    def __init__(self, model_name, model_args,
                 prompt_length, max_length,
                 optimizer,
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
    

    def _shared_eval(self, batch, batch_idx, prefix):
        input_ids, attention_mask = batch

        outputs = self.model(input_ids = input_ids, 
                             attention_mask=attention_mask,
                             labels=input_ids)

        loss = outputs.loss

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
