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
# from transformers import logging
from sklearn.model_selection import StratifiedKFold
from statistics import mean

# logging.set_verbosity_warning()

labels = {'No Disaster': 0, 'Disaster': 1}
debugging = False


def define_objective(project, epochs, notes, seed, device):
    def objective(trial):
        # Read data
        train_data = pd.read_csv('Data/train.csv')
        if debugging:
            train_data = train_data.iloc[:32]
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        n_folds = 4
        batch_size = 2
        skf = StratifiedKFold(n_splits=n_folds)

        learning_rate = trial.suggest_float('learning_rate', low=1e-5, high=1e-1, step=0.001)
        n_linear_layers = trial.suggest_int('n_linear_layers', 1, 4)
        n_neurons = trial.suggest_int('n_neurons', 16, 256, 16)
        model_type = trial.suggest_categorical('model_type', ['MLP', 'LSTM'])

        # Wandb
        name = f'{model_type},layers:{n_linear_layers},neurons:{n_neurons},leaning rate:{learning_rate}'
        config = {'learning_rate': learning_rate, 'seed': seed}
        wandb.init(project=project, entity="nickkaparinos", name=name, config=config, notes=notes, group='',
                   reinit=True)

        training_accuracies = dict()
        training_f1_scores = dict()
        validation_accuracies = dict()
        validation_f1_scores = dict()

        for fold, (train_index, validation_index) in enumerate(skf.split(X_train, y_train)):
            training_accuracies[fold] = []
            training_f1_scores[fold] = []
            validation_accuracies[fold] = []
            validation_f1_scores[fold] = []

            # Split
            X_train_fold, y_train_fold = train_data.iloc[train_index], y_train.iloc[train_index]
            X_val_fold, y_val_fold = train_data.iloc[validation_index], y_train.iloc[validation_index]

            train_dataset, validation_dataset = Dataset(X_train_fold, y_train_fold), Dataset(X_val_fold, y_val_fold)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

            # Model
            model = BertClassifier(n_linear_layers, n_neurons, model_type).to(device)
            loss_fn = torch.nn.NLLLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(epochs):
                print(f'{epoch = }')
                epoch_train_accuracy, epoch_train_f1 = train_one_epoch(model, train_dataloader, loss_fn, optimizer,
                                                                       epoch + 1, fold, device)
                epoch_validation_accuracy, epoch_validation_f1 = evaluation(validation_dataloader, model, loss_fn,
                                                                            epoch + 1, fold, device)

                # Add metrics to dictionaries
                training_accuracies[fold].append(epoch_train_accuracy)
                training_f1_scores[fold].append(epoch_train_f1)
                validation_accuracies[fold].append(epoch_validation_accuracy)
                validation_f1_scores[fold].append(epoch_validation_f1)

        # Log mean epoch metrics
        keys = [i for i in range(n_folds)]
        mean_training_accuracy = [mean([training_accuracies[key][epoch] for key in keys]) for epoch in range(epochs)]
        mean_training_f1_score = [mean([training_f1_scores[key][epoch] for key in keys]) for epoch in range(epochs)]
        mean_validation_accuracy = [mean([validation_accuracies[key][epoch] for key in keys]) for epoch in
                                    range(epochs)]
        mean_validation_f1_score = [mean([validation_f1_scores[key][epoch] for key in keys]) for epoch in range(epochs)]
        metrics = [(mean_training_accuracy, 'Mean Training Accuracy'), (mean_training_f1_score, 'Mean Training F1'),
                   (mean_validation_accuracy, 'Mean Validation Accuracy'),
                   (mean_validation_f1_score, 'Mean Validation F1')]

        for metric, name in metrics:
            for epoch in range(epochs):
                wandb.log({'Epoch': epoch, name: metric[epoch]})

        max_validation_accuracy = max(mean_validation_accuracy)
        wandb.log({'Max Validation Accuracy': max_validation_accuracy})
        return max_validation_accuracy

    return objective


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

    def __init__(self, n_linear_layers=2, n_neurons=32, type='MLP'):
        super(BertClassifier, self).__init__()
        self.type = type
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.lstm = nn.LSTM(768, n_neurons, batch_first=True, bidirectional=True)

        mlp = []
        if n_linear_layers > 1:
            mlp.extend([nn.LazyLinear(n_neurons), nn.ReLU()])
        for _ in range(n_linear_layers - 2):
            mlp.extend([nn.Linear(n_neurons, n_neurons), nn.ReLU()])
        mlp.extend([nn.LazyLinear(2), nn.LogSoftmax(dim=1)])
        self.mlp = nn.Sequential(*mlp)

        self.linear = nn.Linear(768, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_id, mask):
        sequence_output, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        if type == 'LSTM':
            lstm_output, (h, c) = self.lstm(sequence_output)
            n = lstm_output.shape[2]
            x = torch.cat((lstm_output[:, 0, n // 2:], lstm_output[:, -1, :n // 2]), dim=-1)
        else:
            x = pooled_output

        x = self.mlp(x)

        return x


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
        # wandb.log(data={'Epoch': epoch, f'Training Loss {fold}': loss.item()})
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

    wandb.log(data={'Epoch': epoch, f'Training Accuracy {fold}': train_accuracy, f'Training F1 {fold}': train_f1})

    return train_accuracy, train_f1


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
    validation_f1 = f1_score(y_validation, y_pred, average='micro')

    wandb.log(data={'Epoch': epoch, f'Validation Accuracy {fold}': validation_accuracy,
                    f'Validation F1 {fold}': validation_f1})

    return validation_accuracy, validation_f1


def save_dict_to_file(dictionary, path, txt_name='hyperparameter_dict'):
    with open(f'{path}/{txt_name}.txt', 'w') as f:
        f.write(str(dictionary))


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
