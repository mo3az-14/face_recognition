import torch
import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from copy import deepcopy
import torch.nn as nn
import torch.cuda.amp as amp


def get_metrics(
    probs: torch.tensor, targets: torch.tensor, thresholds: list[float], output
) -> dict:
    for i in thresholds:
        predictions = (probs > i).astype(int)
        prec, rec, f, _ = precision_recall_fscore_support(targets, predictions)
        acc = predictions == targets
        acc = acc.sum() / len(targets)
        output["accuracy"].append(acc)
        output["precision"].append(prec[0])
        output["recall"].append(rec[0])
        output["fscore"].append(f[0])

    return output


# training
def train_step(
    model: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    loss_fn: torch.nn,
    optimizer: torch.optim,
    mixed_precision_on: bool = True,
    device: str = "cuda",
    scaler=None,
    calculate_accuracy: bool = False,
):
    model.train()
    train_loss: float = 0.0

    probs = []
    targets = []
    for first, second, target in tqdm.tqdm(train_data):
        optimizer.zero_grad()
        first, second, target = (
            first.to(device, non_blocking=True),
            second.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
        print(first.shape)
        if mixed_precision_on:
            with torch.autocast(device):
                output = model(first, second).squeeze()
                loss = loss_fn(output, target)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()
        else:
            output = model(first, second).squeeze()

            loss = loss_fn(output, target)

            loss.backward()

            optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            if calculate_accuracy:
                probs.append(torch.sigmoid(output).cpu().numpy())
                targets.append(target.squeeze().cpu().numpy())

    return train_loss, probs, targets


# validation
def valid_step(
    model: torch.nn.Module,
    test_data: torch.utils.data.DataLoader,
    loss_fn: torch.nn,
    device: str,
    mixed_precision_on: bool = True,
    calculate_accuracy: bool = False,
):
    test_loss = 0.0

    probs = []
    targets = []
    with torch.no_grad():
        model.eval()
        for first, second, target in tqdm.tqdm(test_data):
            first, second, target = (
                first.to(device, non_blocking=True),
                second.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            if mixed_precision_on:
                # saves memory
                with torch.autocast(device):
                    output = model(first, second).squeeze()
                    loss = loss_fn(output, target)

            else:
                output = model(first, second).squeeze()
                loss = loss_fn(output, target)

            if calculate_accuracy:
                probs.append(torch.sigmoid(output).cpu().numpy())
                targets.append(target.squeeze().cpu().numpy())

            test_loss += loss.item()

    return test_loss, probs, targets


# training loop
def train_loop(
    model: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    test_data: torch.utils.data.DataLoader,
    loss_fn: torch.nn,
    optimizer: torch.optim,
    device: str = "cuda",
    epochs: int = 5,
    patience: int = 20,
    lr_scheduler: torch.optim.lr_scheduler = None,
    mixed_precision_on: bool = True,
    calc_metrics_interval: int = None,
    early_stopping_metric: str = None,
):
    train_loss_acc = []
    test_loss_acc = []
    scaler = None
    if mixed_precision_on:
        scaler = torch.GradScaler("cuda")

    best_loss = float("inf")
    epochs_without_imporvement = 0
    best_accuracy, best_precision, best_recall = 0, 0, 0

    train_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "fscore": [],
    }
    test_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "fscore": [],
    }

    for i in range(epochs):
        calculate_metrics = (
            calc_metrics_interval is not None and (i % calc_metrics_interval) == 0
        )
        # triaing step
        train_loss, train_probs, train_targets = train_step(
            model,
            train_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            mixed_precision_on=mixed_precision_on,
            scaler=scaler,
            calculate_accuracy=calculate_metrics,
        )
        # validation step
        test_loss, test_probs, test_targets = valid_step(
            model, test_data, loss_fn, device, calculate_accuracy=calculate_metrics
        )

        # learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        if calculate_metrics:
            train_probs, train_targets = (
                np.concatenate(train_probs),
                np.concatenate(train_targets),
            )
            test_probs, test_targets = (
                np.concatenate(test_probs),
                np.concatenate(test_targets),
            )
            train_metrics = get_metrics(
                train_probs, train_targets, [0.5], output=train_metrics
            )
            test_metrics = get_metrics(
                test_probs, test_targets, [0.5], output=test_metrics
            )
            del (train_probs, train_targets)
        if early_stopping_metric == "train":
            if train_loss < best_loss:
                best_loss = train_loss
                epochs_without_imporvement = 0
                best_loss_model = deepcopy(model.state_dict())
            else:
                epochs_without_imporvement += 1
        else:
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_without_imporvement = 0
                best_loss_model = deepcopy(model.state_dict())
            else:
                epochs_without_imporvement += 1

        if best_accuracy < test_metrics["accuracy"][-1]:
            best_accuracy_model = deepcopy(model.state_dict())
        if best_precision < test_metrics["precision"][-1]:
            best_precision_model = deepcopy(model.state_dict())
        if best_recall < test_metrics["recall"][-1]:
            best_recall_model = deepcopy(model.state_dict())

        train_loss_acc.append(train_loss)
        test_loss_acc.append(test_loss)

        print(
            f"train loss: {train_loss} test loss: {test_loss} train accuracy: {train_metrics['accuracy'][-1]} test accuracy: {test_metrics['accuracy'][-1]} @ epoch {i}"
        )

        if epochs_without_imporvement >= patience and patience != 0:
            models = {
                "best_loss_model": best_loss_model,
                "best_accuracy_model": best_accuracy_model,
                "best_precision_model": best_precision_model,
                "best_recall_model": best_recall_model,
            }
            print("early stopping activated ")
            return (train_loss_acc, test_loss_acc, train_metrics, test_metrics, models)

    models = {
        "best_loss_model": best_loss_model,
        "best_accuracy_model": best_accuracy_model,
        "best_precision_model": best_precision_model,
        "best_recall_model": best_recall_model,
    }
    return (train_loss_acc, test_loss_acc, train_metrics, test_metrics, models)


# intializing weights
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
