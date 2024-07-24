import torch
import numpy as np
from torch import nn
import torch.utils
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import torch.utils.data
import torchvision.transforms.v2 as transforms
import torch.cuda.amp as amp
import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import config
from data_loaders import Pair_Data_Loader
from model import Model
import copy
import my_logger as log
from arguments import get_arguments


def get_metrics(
    probs: torch.tensor, targets: torch.tensor, thresholds: list[float], output
) -> dict:
    for i in thresholds:
        predictions = (probs > i).astype(int)
        prec, rec, f, _ = precision_recall_fscore_support(targets, predictions)
        acc = predictions == targets
        acc = acc.sum() / len(targets)
        output["accuracy"].append(acc)
        output["precision"].append(prec)
        output["recall"].append(rec)
        output["fscore"].append(f)

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
):
    model.train()
    train_loss: float = 0.0

    for first, second, target in tqdm.tqdm(train_data):
        optimizer.zero_grad()
        first, second, target = (
            first.to(device, non_blocking=True),
            second.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
        if mixed_precision_on:
            with amp.autocast():
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

        train_loss += loss.item()

    return train_loss


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
                with amp.autocast():
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
    accuracy_interval: int = None,
    early_stopping_metric: str = None,
):
    train_loss_acc = []
    test_loss_acc = []
    scaler = None
    if mixed_precision_on:
        scaler = amp.GradScaler()

    if patience is not None:
        best_loss = float("inf")
        epochs_without_imporvement = 0

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "fscore": [],
    }

    for i in range(epochs):
        accuracy_on = accuracy_interval is not None and (i % accuracy_interval) == 0
        # triaing step
        train_loss = train_step(
            model,
            train_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            mixed_precision_on=mixed_precision_on,
            scaler=scaler,
        )
        # validation step
        test_loss, probs, targets = valid_step(
            model, test_data, loss_fn, device, calculate_accuracy=accuracy_on
        )
        # early stopping
        print(f"train loss: {train_loss} test loss: {test_loss} @ epoch {i}")
        if patience is not None:
            if early_stopping_metric == "train":
                if train_loss < best_loss:
                    best_loss = train_loss
                    epochs_without_imporvement = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    epochs_without_imporvement += 1

            else:
                if test_loss < best_loss:
                    best_loss = test_loss
                    epochs_without_imporvement = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    epochs_without_imporvement += 1

            if epochs_without_imporvement >= patience:
                print("early stopping activated ")
                break

        # learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        if accuracy_on:
            probs, targets = np.concatenate(probs), np.concatenate(targets)
            metrics = get_metrics(probs, targets, [0.5], output=metrics)

        train_loss_acc.append(train_loss)
        test_loss_acc.append(test_loss)

    if patience is not None:
        model.load_state_dict(best_model_wts)

    return (train_loss_acc, test_loss_acc, metrics)


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


if __name__ == "__main__":
    # get arguments from the console. defaults are in arguments.py
    args = get_arguments()
    lr = args.lr
    weight_decay = args.weight_decay
    gamma = args.gamma
    adam = args.adam
    init = args.init
    cos = args.cos
    step_size = args.step_size
    lr_scheduler_on = args.lr_scheduler_on
    mixed_precision = args.mixed_precision
    batch_size = args.batch_size
    patience = args.patience
    epochs = args.epochs
    num_workers = args.num_workers
    accuracy_interval = args.accuracy_interval
    slice_of_data = args.slice_of_data
    early_stopping_metric = args.early_stopping_metric
    p = args.p
    torch.backends.cudnn.benchmark = True

    rng = np.random.default_rng()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model().to(device)

    optimizer = (
        Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
        if adam
        else SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    )

    # learning rate scheduler
    scheduler = None
    if lr_scheduler_on:
        if cos:
            CosineAnnealingLR(optimizer, T_max=50)
        else:
            StepLR(optimizer, step_size=step_size, gamma=gamma)

    # transformations
    data_transform = data_transform = nn.Sequential(
        transforms.ToImage(),
        transforms.ToDtype(
            torch.float32,
            scale=True,
        ),
        transforms.Resize(size=config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=p),
        transforms.RandomVerticalFlip(p=p),
        transforms.RandomErasing(p=p),
        transforms.RandomAffine(180),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )

    # data stuff
    pair_data_train = Pair_Data_Loader(
        root=config.TRAIN_DATASET,
        transform=data_transform,
        indices=np.arange(slice_of_data) if slice_of_data else None,
    )
    pair_dataloader_train = DataLoader(
        pair_data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    pair_data_test = Pair_Data_Loader(
        root=config.TEST_DATASET,
        transform=data_transform,
        indices=np.arange(slice_of_data) if slice_of_data else None,
    )
    pair_dataloader_test = DataLoader(
        pair_data_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    loss_function = nn.BCEWithLogitsLoss(reduction="mean")

    # weights intialization
    if init == 1:
        model.apply(initialize_weights)

    # training
    train_loss, test_loss, metrics = train_loop(
        model,
        pair_dataloader_train,
        pair_dataloader_test,
        loss_function,
        optimizer,
        patience=patience,
        device=device,
        epochs=epochs,
        lr_scheduler=scheduler,
        mixed_precision_on=mixed_precision,
        accuracy_interval=accuracy_interval,
        early_stopping_metric=early_stopping_metric,
    )

    print(f"Final train loss: {train_loss} , Final test loss: {test_loss}")
    # saving settings and model
    id = log.gen_id()
    saving_path = log.make_dir(id)
    params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "lr": lr,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "step_size": step_size,
        "lr_scheduler_on": lr_scheduler_on,
        "mixed_precision": mixed_precision,
        "batch_size": batch_size,
        "patience": patience,
        "epochs": epochs,
        "num_workers": num_workers,
        "accuracy_interval": accuracy_interval,
        "slice_of_data": slice_of_data,
        "early_stopping_metric": early_stopping_metric,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "fscore": metrics["fscore"],
    }
    torch.save(params, saving_path)
    print(f"model saved in {saving_path}")
