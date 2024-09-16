# model.py
import pdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict

from dataset_classes.datasets import DatasetGenerator, collate_fn
from dataset_classes.preprocess import preprocess_all_files
from models.whisper import *

class Classifier(pl.LightningModule):
    """ 
    AutoEncoder model class using PyTorch Lightning.

    Args:
        hparams (Dict): Dictionary of hyperparameters.
        files (Dict): Dictionary of files.
    """
    def __init__(self, hparams: Dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = WhisperWrapper(pretrained_model=self.hparams.embedding, lora_rank=self.hparams.lora_rank)
        self.train_files, self.val_files = preprocess_all_files(hparams)
        self.test_files = self.val_files
        self.train_val_test_split()
        self.criterion = nn.NLLLoss()
        self.prediction_dict = defaultdict(list)
        self.training_save, self.validation_save, self.testing_save = [], [], []

    def training_step(self, batch, batch_idx):
        # read data
        x, y = batch
        x = x.float()
        # forward pass
        outputs = self.model(x)
        # backword pass
        outputs = torch.log_softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, y)
        # log loss
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        # score = f1_score(y, y_pred, average="macro")
        # self.log("train_f1_score", score)
        self.training_save.append({"loss": loss, "pred": pred.detach().cpu(), "y": y.detach().cpu()})
        return loss

    def validation_step(self, batch, batch_idx):
        # read data
        x, y = batch
        x = x.float()
        # forward pass
        outputs = self.model(x)

        # backword pass
        outputs = torch.log_softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, y)
        # log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.validation_save.append({"loss": loss, "pred": pred.detach().cpu(), "y": y.detach().cpu()})
        return loss

    def test_step(self, batch, batch_idx):
        # read data
        x, y, file_name, start = batch
        x = x.float()
        # forward pass
        outputs = self.model(x)
        # log loss
        outputs = torch.log_softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1).detach().cpu()
        loss = self.criterion(outputs, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        # self.log_dict({"model_weights": self.model.weights}, on_step=False, on_epoch=True)
        self.testing_save.append({"loss": loss.detach().cpu(), "pred": pred.detach().cpu(), "y": y.detach().cpu(), "file_name": file_name, "start": start.detach().cpu()})

        return loss
    
    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError("Invalid optimizer.")
        return optimizer

    
    def on_validation_epoch_end(self):
        y_pred =  torch.cat([x["pred"]for x in self.validation_save])
        y =  torch.cat([x["y"]for x in self.validation_save])
        accuracies = []
        for y_pred, y in zip(y_pred, y):
            accuracies.append(accuracy_score(y, y_pred))
        mean_acc = sum(accuracies) / len(accuracies)
        self.log('val_score', mean_acc)
        self.validation_save = []

    def on_test_epoch_end(self):
        if not self.testing_save:
            return
        
        # log score
        y_pred =  torch.cat([x["pred"]for x in self.testing_save]).numpy()
        y =  torch.cat([x["y"]for x in self.testing_save]).numpy()
        accuracies = []
        for y_pred, y in zip(y_pred, y):
            accuracies.append(accuracy_score(y, y_pred))
        mean_acc = sum(accuracies) / len(accuracies)
        self.log('test_score', mean_acc)
        self.testing_save = []

    
    def train_dataloader(self):
        
        data_generator = DatasetGenerator(
            data_list=self.train_data, 
            data_len=len(self.train_data),
            is_test=False
        )
        train_loader = DataLoader(
            data_generator, 
            batch_size=self.hparams.batch_size, 
            num_workers=5, 
            shuffle=True,
            # collate_fn=collate_fn,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        data_generator = DatasetGenerator(
            data_list=self.val_data, 
            data_len=len(self.val_data),
            is_test=False
        )
        return DataLoader(
            data_generator, 
            batch_size=self.hparams.batch_size, 
            num_workers=5, 
            # collate_fn=collate_fn,
            shuffle=False,
            drop_last=False
        )
    
    def test_dataloader(self):
        data_generator = DatasetGenerator(
            data_list=self.test_data, 
            data_len=len(self.test_data),
            is_test=True
        )
        test_dataloader = DataLoader(
            data_generator, 
            batch_size=1, 
            num_workers=5, 
            shuffle=False,
            drop_last=False
        )

        return test_dataloader
    
    def on_save_checkpoint(self, checkpoint):
        if "whisper" in self.hparams.embedding:
            checkpoint['embed_pos_weight'] = self.model.backbone_model.encoder.embed_positions.weight

    def on_load_checkpoint(self, checkpoint):
        if "whisper" in self.hparams.embedding:
            self.model.backbone_model.encoder.embed_positions.weight = checkpoint['embed_pos_weight']

    def train_val_test_split(self):
        self.train_data = self.return_data(self.train_files, True)
        self.val_data = self.return_data(self.val_files)
        self.test_data = self.return_data(self.test_files)
        

        # # for debug
        if self.hparams.debug:
            self.train_data = self.train_data[:200]
            self.val_data = self.val_data[:200]
            self.test_data = self.test_data[:10]



    def return_data(self, files, istrain = False):
        """
        return data for the class0/class1 classifier

        args:
            files: a dictionary of files
        return:
            data: a list of data points
        """
        data = []
        for k, segments in files.items():
            segments_len = len(segments)
            for segment in segments:
                data.append({"audio": segment["audio"], "labels": segment["labels"], "start": segment["start"], "file_name": k})
        
        return data
    


class Classifier_eval(pl.LightningModule):
    """ 
    AutoEncoder model class using PyTorch Lightning.

    Args:
        hparams (Dict): Dictionary of hyperparameters.
        files (Dict): Dictionary of files.
    """
    def __init__(self, hparams: Dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = WhisperWrapper(pretrained_model=self.hparams.embedding, lora_rank=self.hparams.lora_rank)

    def training_step(self, batch, batch_idx):
        # read data
        x, y = batch
        # if self.hparams.embedding == "wavlm":
        #     y = y[:, :-1]   # remove the last token for wavlm
        x = x.float()
        # forward pass
        outputs = self.model(x)
        # backword pass
        outputs = torch.log_softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, y)
        # log loss
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        # score = f1_score(y, y_pred, average="macro")
        # self.log("train_f1_score", score)
        self.training_save.append({"loss": loss, "pred": pred.detach().cpu(), "y": y.detach().cpu()})
        return loss

    def validation_step(self, batch, batch_idx):
        # read data
        x, y = batch
        # if self.hparams.embedding == "wavlm":
        #     y = y[:, :-1]   # remove the last token for wavlm
        x = x.float()
        # forward pass
        outputs = self.model(x)
        # backword pass
        outputs = torch.log_softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, y)
        # log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.validation_save.append({"loss": loss, "pred": pred.detach().cpu(), "y": y.detach().cpu()})
        return loss

    def test_step(self, batch, batch_idx):
        # read data
        x, y, file_name, start = batch
        # if self.hparams.embedding == "wavlm":
        #     y = y[:, :-1]   # remove the last token for wavlm
        x = x.float()
        # forward pass
        outputs = self.model(x)
        # log loss
        outputs = torch.log_softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1).detach().cpu()
        loss = self.criterion(outputs, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        # self.log_dict({"model_weights": self.model.weights}, on_step=False, on_epoch=True)
        self.testing_save.append({"loss": loss.detach().cpu(), "pred": pred.detach().cpu(), "y": y.detach().cpu(), "file_name": file_name, "start": start.detach().cpu()})

        return loss
    
    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError("Invalid optimizer.")
        return optimizer

    
    def on_validation_epoch_end(self):
        y_pred =  torch.cat([x["pred"]for x in self.validation_save])
        y =  torch.cat([x["y"]for x in self.validation_save])
        accuracies = []
        for y_pred, y in zip(y_pred, y):
            accuracies.append(accuracy_score(y, y_pred))
        mean_acc = sum(accuracies) / len(accuracies)
        self.log('val_score', mean_acc)
        self.validation_save = []

    def on_test_epoch_end(self):
        if not self.testing_save:
            return
        
        # log score
        y_pred =  torch.cat([x["pred"]for x in self.testing_save]).numpy()
        y =  torch.cat([x["y"]for x in self.testing_save]).numpy()
        accuracies = []
        for y_pred, y in zip(y_pred, y):
            accuracies.append(accuracy_score(y, y_pred))
        mean_acc = sum(accuracies) / len(accuracies)
        self.log('test_score', mean_acc)
        self.testing_save = []

    
    def train_dataloader(self):
        
        data_generator = DatasetGenerator(
            data_list=self.train_data, 
            data_len=len(self.train_data),
            is_test=False
        )
        train_loader = DataLoader(
            data_generator, 
            batch_size=self.hparams.batch_size, 
            num_workers=5, 
            shuffle=True,
            # collate_fn=collate_fn,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        data_generator = DatasetGenerator(
            data_list=self.val_data, 
            data_len=len(self.val_data),
            is_test=False
        )
        return DataLoader(
            data_generator, 
            batch_size=self.hparams.batch_size, 
            num_workers=5, 
            # collate_fn=collate_fn,
            shuffle=False,
            drop_last=False
        )
    
    def test_dataloader(self):
        data_generator = DatasetGenerator(
            data_list=self.test_data, 
            data_len=len(self.test_data),
            is_test=True
        )
        test_dataloader = DataLoader(
            data_generator, 
            batch_size=1, 
            num_workers=5, 
            shuffle=False,
            drop_last=False
        )

        return test_dataloader
    
    def on_save_checkpoint(self, checkpoint):
        if "whisper" in self.hparams.embedding:
            checkpoint['embed_pos_weight'] = self.model.backbone_model.encoder.embed_positions.weight

    def on_load_checkpoint(self, checkpoint):
        if "whisper" in self.hparams.embedding:
            self.model.backbone_model.encoder.embed_positions.weight = checkpoint['embed_pos_weight']

    def train_val_test_split(self):
        self.train_data = self.return_data(self.train_files, True)
        self.val_data = self.return_data(self.val_files)
        self.test_data = self.return_data(self.test_files)
        

        # for debug
        if self.hparams.debug:
            self.train_data = self.train_data[:100]
            self.val_data = self.val_data[:10]
            self.test_data = self.test_data[:10]



    def return_data(self, files, istrain = False):
        """
        return data for the class0/class1 classifier

        args:
            files: a dictionary of files
        return:
            data: a list of data points
        """
        data = []
        for k, segments in files.items():
            segments_len = len(segments)
            # if istrain:
            #     for segment in segments[:round(segments_len * self.hparams.percent)]:
            #         data.append({"audio": segment["audio"], "labels": segment["labels"], "start": segment["start"], "file_name": k})
            # else:
            #     for segment in segments:
            #         data.append({"audio": segment["audio"], "labels": segment["labels"], "start": segment["start"], "file_name": k})
            for segment in segments:
                data.append({"audio": segment["audio"], "labels": segment["labels"], "start": segment["start"], "file_name": k})
        
        return data

