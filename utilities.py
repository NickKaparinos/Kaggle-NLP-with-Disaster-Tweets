"""
Natural Language Processing with Disaster Tweets
Kaggle competition
Nick Kaparinos
2022
"""

import random
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from transformers import logging

logging.set_verbosity_warning()

labels = {'No Disaster': 0, 'Disaster': 1}


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, y):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = y
        self.texts = [self.tokenizer(text, padding='max_length', max_length=32, truncation=True, return_tensors="pt")
                      for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels.iloc[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train_one_epoch(model, dataloader, loss_fn, optimizer, epoch, fold, device):
    running_loss = 0.
    model.train()
    y_train = np.empty((0,))
    y_pred = np.empty((0,))

    for batch_num, batch in enumerate(tqdm(dataloader)):
        x_batch, y_batch = batch[0], batch[1].to(device)

        mask = x_batch['attention_mask'].to(device)
        input_id = x_batch['input_ids'].squeeze(1).to(device)

        # Inference
        output = model(input_id, mask)
        y_pred_batch = torch.argmax(output.cpu(), dim=1).numpy()

        #  Loss
        optimizer.zero_grad()
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_num % 10 == 9:
            avg_loss = running_loss / 10
            running_loss = 0.
            wandb.log(data={'Epoch': epoch, f'Training Loss {fold}': avg_loss})

        # Stack
        y_train = np.hstack([y_train, y_batch.cpu().numpy()]) if y_train.size else y_batch.cpu().numpy()
        y_pred = np.hstack([y_pred, y_pred_batch]) if y_pred.size else y_pred_batch

    # Training Metrics
    train_accuracy = accuracy_score(y_train, y_pred)
    train_f1 = f1_score(y_train, y_pred, average='micro')

    wandb.log(data={'Epoch': epoch, 'Training Accuracy': train_accuracy, 'Training F1': train_f1})

    return train_accuracy


def evaluation(dataloader, model, loss_fn, epoch, fold, device):
    model.eval()
    running_loss = 0.

    with torch.no_grad():
        y_validation = np.empty((0,))
        y_pred = np.empty((0,))
        for batch_num, batch in enumerate(tqdm(dataloader)):
            x_batch, y_batch = batch[0], batch[1].to(device)

            mask = x_batch['attention_mask'].to(device)
            input_id = x_batch['input_ids'].squeeze(1).to(device)

            # Inference
            output = model(input_id, mask)
            y_pred_batch = torch.argmax(output.cpu(), dim=1).numpy()

            #  Loss
            loss = loss_fn(output, y_batch)

            running_loss += loss.item()
            if batch_num % 10 == 9:
                avg_loss = running_loss / 10
                running_loss = 0.
                wandb.log(data={'Epoch': epoch, f'Validation Loss {fold}': avg_loss})

            # Stack
            y_validation = np.hstack(
                [y_validation, y_batch.cpu().numpy()]) if y_validation.size else y_batch.cpu().numpy()
            y_pred = np.hstack([y_pred, y_pred_batch]) if y_pred.size else y_pred_batch

    # Validation Metrics
    validation_accuracy = accuracy_score(y_validation, y_pred)
    train_f1 = f1_score(y_validation, y_pred, average='micro')

    wandb.log(data={'Epoch': epoch, 'Validation Accuracy': validation_accuracy, 'Training F1': train_f1})

    return validation_accuracy


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
