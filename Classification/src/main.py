import argparse
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from para import Hyperparameter
from utils import load_data, enumerateWord
from models import RNN_LSTM, RNN_GRU, CNN, MLP


config = Hyperparameter()


def get_data():
    word2id = enumerateWord()
    train_contents, train_labels = load_data("../Dataset/train.txt", max_length, word2id)
    val_contents, val_labels = load_data("../Dataset/validation.txt", max_length, word2id)
    test_contents, test_labels = load_data("../Dataset/test.txt", max_length, word2id)
    train_dataset = TensorDataset(
        torch.from_numpy(train_contents).type(torch.float),
        torch.from_numpy(train_labels).type(torch.long),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_contents).type(torch.float),
        torch.from_numpy(val_labels).type(torch.long),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_contents).type(torch.float),
        torch.from_numpy(test_labels).type(torch.long),
    )
    return (DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2))


def parser_data():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-3,
        help="The initial learning rate.",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        dest="epoch",
        type=int,
        default=10,
        help="Epoch"
    )
    parser.add_argument(
        "-m",
        "--max_length",
        dest="max_length",
        type=int,
        default=120,
        help="The maximum length of the sequence.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=50,
        help="batch size"
    )
    parser.add_argument(
        "-c",
        "--choice",
        dest="choice",
        type=str,
        default="CNN",
        help="Choice of model: CNN, RNN_LTSM, RNN_GRU, MLP.",
    )
    args = parser.parse_args()
    selection = args.choice
    if selection == "CNN":
        selected_model = CNN(config).to(DEVICE)
    elif selection == "RNN_LSTM":
        selected_model = RNN_LSTM(config).to(DEVICE)
    elif selection == "RNN_GRU":
        selected_model = RNN_GRU(config).to(DEVICE)
    elif selection == "MLP":
        selected_model = MLP(config).to(DEVICE)
    else:
        print("Please select one of the following: CNN, RNN_LTSM, RNN_GRU, MLP.")
        exit(1)
    return args.learning_rate, args.epoch, args.max_length, args.batch_size, selected_model


def eval_model(dataloader, is_training):
    loss, acc = 0.0, 0.0
    count, correct = 0, 0
    full_true = []
    full_pred = []

    for (x, y) in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        batch_loss = criterion(output, y)
        loss += batch_loss.item()
        correct += (output.argmax(1) == y).to(torch.float32).sum().item()
        count += len(x)
        full_true.extend(y.cpu().numpy().tolist())
        full_pred.extend(output.argmax(1).cpu().numpy().tolist())

        if is_training:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    loss *= dataloader.batch_size
    loss /= len(dataloader.dataset)
    acc = correct / count
    f_score = f1_score(np.array(full_true), np.array(full_pred), average="binary")

    if is_training:
        scheduler.step()

    return loss, acc, f_score


def train(dataloader):
    model.train()
    return eval_model(dataloader, is_training=True)


def valid_and_test(dataloader):
    model.eval()
    return eval_model(dataloader, is_training=False)


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate, epoch, max_length, batch_size, model = parser_data()
    train_dataloader, val_dataloader, test_dataloader = get_data()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5)

    wandb.init(project=f"Classification", name=f"{model.__name__}", entity="leverimmy")
    wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 50}

    for each in tqdm(range(1, epoch + 1)):
        train_loss, train_acc, train_f_score = train(train_dataloader)
        val_loss, val_acc, val_f_score = valid_and_test(val_dataloader)
        test_loss, test_acc, test_f_score = valid_and_test(test_dataloader)
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f_score": train_f_score,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f_score1": val_f_score,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f_score": test_f_score,
            }
        )
        print(
            f'Epoch {each}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, '
            f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, '
            f'test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}'
        )
