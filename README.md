# Face recogntion by moaaz

This project is an attempt to replicate
[Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf "Siamese Neural Networks for One-shot Image Recognition")
paper with some changes like using the [lfw](https://www.kaggle.com/datasets/atulanandjha/lfwpeople "LFW from kaggle") dataset instead of the
[omniglot](https://www.kaggle.com/datasets/qweenink/omniglot "omniglot from kaggle") dataset and using Adam optimizer instead of SGD (I tried it but adam converged faster ).
The goal of the project is to match the faces of people (a very naive face recognition/lock for the phones).The main purpose of this
project to me was **learning**.I learned a lot about pytorch (90%), numpy (10%) and computer vision in general in this project.
My goal was to get my hands dirty with pytorch and try replicating a paper.

## How to download the data

1- go to kaggle and create an API token ([docs](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication "kaggle docs"))

2- copy the token to the `config.py`

```
KAGGLE_USERNAME = "your username"
KAGGLE_KEY = "your token"
```

3- run

```
python download_data.py
```

## train the model

to train the model run

```
python trainer.py -[Arguments]
```

we can choose between

- 2 optimizers Adam and SGD using `--adam` argument
- 2 learning rate schedulers StepLr and CosineAnnealingLR

### Arguments

--epochs => No. of epochs for training. default 5

--batch_size => Batch size . default 128

--num_workers => No. of CPU cores used in dataloaders. default 0

--lr => Learning rate. default 0.0001

--lr_scheduler_on => Turn learning rate scheduler on or off. default: 1 (0 or 1)

--gamma => Multiplicative factor of learning rate decay for StepLR. default 0.999

--step_size => Period of learning rate decay for StepLR. default 1

--weight_decay => Weight decay (L2 penalty). default: 0.01

--mixed_precision => use mixed precision training. default: 1 (0 or 1)

--patience => No. of epochs to wait if there is no improvement in the early stoping metric defaut: 0 (early stopping is off)

--calc_metrics_interval => Period to calculate performance metrics. default: 1

--slice_of_data => train on a subset (slice) of data. default: None (test)

--early_stopping_metric => which loss to use for early stopping "train" or "test".

--init => use weight intialization. default 1

--cos => use CosineAnnealingLR or StepLR. 1->CosineAnnealingLR and 0-> StepLR. default 1

--p => probalitiy for data augmentation. default 0.5

--adam => use Adam optimizer or SGD. 1->Adam and 0->SGD
