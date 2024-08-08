import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="parameteres for training")

    parser.add_argument("--epochs", type=int, default=5, help="epochs. default 5")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size. default 128"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers for data loader. default 0",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for the model. default 0.0001",
    )
    parser.add_argument(
        "--lr_scheduler_on",
        type=int,
        default=1,
        help="Learning rate for the model. default 1",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="gamma for the learning rate scheduler. default 0.999",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="step size for learning rate scheduler. default 1",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="weight decay. default 0.01"
    )
    parser.add_argument(
        "--mixed_precision",
        type=int,
        default=1,
        help="enable mixed precision. default True",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="patience for early stopping. default 20",
    )
    parser.add_argument(
        "--calc_metrics_interval",
        type=int,
        default=1,
        help="interval to calculate accuracy. default 1",
    )
    parser.add_argument(
        "--slice_of_data",
        type=int,
        default=None,
        help="gets only a slice of data. default is None which is all data. default None",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default=None,
        help="metric for early stopping. default test",
    )
    parser.add_argument(
        "--init",
        type=int,
        default=1,
        help="command for enabling or disabling the intialization",
    )
    parser.add_argument(
        "--cos",
        type=int,
        default=1,
        help="1-> cos annealing LR scheduler 0-> Step Lr Scheduler (spicify gamma and step size. default : 0.99 and 1)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="p for data augmentation",
    )
    parser.add_argument("--adam", type=int, default=1, help="1 -> adam , 0-> SGD")
    parser.add_argument(
        "--siamese",
        type=int,
        default=1,
        help="choose between the original siamese neural network or resnext as a backbone siamese. default 1-> siamese",
    )
    args = parser.parse_args()

    return args
