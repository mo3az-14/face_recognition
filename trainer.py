import torch
import numpy as np
from torch import nn
import torch.utils
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import torch.utils.data
import torchvision.transforms.v2 as transforms
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import config
from data_loaders import Pair_Data_Loader
from model import original_siamese, resnext_50
import my_logger as log
from arguments import get_arguments
from helper_functions import train_loop, initialize_weights


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
    siamese = args.siamese
    num_workers = args.num_workers
    calc_metrics_interval = args.calc_metrics_interval
    slice_of_data = args.slice_of_data
    early_stopping_metric = args.early_stopping_metric
    p = args.p
    torch.backends.cudnn.benchmark = True

    rng = np.random.default_rng()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if siamese:
        model = original_siamese().to(device)
    else:
        model = resnext_50().to(device)

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
    train_loss, test_loss, train_metrics, test_metrics, models = train_loop(
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
        calc_metrics_interval=calc_metrics_interval,
        early_stopping_metric=early_stopping_metric,
    )

    id = log.gen_id()
    # saving settings and model
    saving_path = log.make_dir(id)
    params = {
        "model": models,
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
        "calc_metrics_interval": calc_metrics_interval,
        "slice_of_data": slice_of_data,
        "early_stopping_metric": early_stopping_metric,
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_fscore": train_metrics["fscore"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_fscore": test_metrics["fscore"],
    }
    torch.save(params, saving_path)
    print(f"model saved in {saving_path}")
