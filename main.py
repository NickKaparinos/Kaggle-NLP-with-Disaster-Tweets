"""
Natural Language Processing with Disaster Tweets
Kaggle competition
Nick Kaparinos
2022
"""

from utilities import *
import pandas as pd
import time
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
import torch


def main():
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    # Read data
    train_data = pd.read_csv('Data/train.csv')
    train_data = train_data.iloc[:25]
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    skf = StratifiedKFold(n_splits=4)

    # Wandb
    project = 'Kaggle-Disaster'
    name = 'tasos'
    notes = ''
    config = {}
    wandb.init(project=project, entity="nickkaparinos", name=name, config=config, notes=notes, group='', reinit=True)

    for train_index, validation_index in skf.split(X_train, y_train):
        # Split
        X_train_fold, y_train_fold = train_data.iloc[train_index], y_train.iloc[train_index]
        X_val_fold, y_val_fold = train_data.iloc[validation_index], y_train.iloc[validation_index]

        train_dataset, validation_dataset = Dataset(X_train_fold, y_train_fold), Dataset(X_val_fold, y_val_fold)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=2)

        # Model
        model = BertClassifier().to(device)
        loss_fn = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 1
        fold = 0

        for epoch in range(epochs):
            print(f'{epoch = }')
            train_one_epoch(model, train_dataloader, loss_fn, optimizer, epoch + 1, fold, device)
            evaluation(validation_dataloader, model, loss_fn, epoch, fold, device)
    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()
