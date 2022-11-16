import pytorch_lightning as pl
import torch
from models.VIT import VIT
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler, SGD
from torch import argmax, tensor, stack
from utils.ResidualNet import Res34
import argparse


# from numpy import mean

class VIT_lightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = Res34(args,3,2)
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
        self.loss = CrossEntropyLoss()

    def forward(self, data_x):
        return self.model(data_x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), self.hparams.learning_rate, weight_decay=1e-5)
        # optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=0.1)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        data_x, data_y = batch

        outputs = self(data_x)
        loss = self.loss(outputs, data_y)
        with torch.no_grad():
            train_pred = argmax(outputs, dim=1)
            train_acc = train_pred == data_y
        return {'loss': loss, 'acc': train_acc}

    def training_step_end(self, step_output):
        accurate = step_output['acc']
        acc = sum(accurate) / len(accurate)
        loss = step_output['loss']
        self.log('training_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_x, data_y = batch
        outputs = self(data_x)
        train_pred = argmax(outputs, dim=1)
        train_acc = train_pred == data_y
        loss = self.loss(outputs, data_y)
        # self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log('val_acc', acurate, on_step=True, on_epoch=False, prog_bar=True)
        return {'val_loss': loss, 'val_acc': train_acc}

    def validation_step_end(self, validation_step_outputs):
        val_out = validation_step_outputs['val_acc']
        acc = sum(val_out) / len(val_out)
        self.log('val_total_acc', acc, on_step=False, on_epoch=True, prog_bar=True)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VIT_lightning")
        parser.add_argument('-e', '--epochs', type=int, default=5,
                            help='input training epoch for training (default: 5)')
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                            help='input learning rate for training (default: 5e-4)')
        return parent_parser
