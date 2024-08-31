## Training script arguments refrence

| Parameter                 | Type    | Default | Description                                                                                              |
| :------------------------ | :------ | :------ | :------------------------------------------------------------------------------------------------------- |
| `--epochs`                | `int`   | 5       | epochs                                                                                                   |
| `--batch_size`            | `int`   | 128     | batch size                                                                                               |
| `--num_workers`           | `int`   | 0       | num workers for data loader                                                                              |
| `--lr`                    | `float` | 0.0001  | learning rate                                                                                            |
| `--lr_scheduler_on`       | `int`   | 1       | enable/disable lr scheduler                                                                              |
| `--gamma`                 | `float` | 0.999   | gamma for StepLR scheduler. new lr = old lr \* gamma                                                     |
| `--step_size`             | `int`   | 1       | step size for learning rate scheduler                                                                    |
| `--weight_decay`          | `float` | 0.01    | weight decay                                                                                             |
| `--mixed_precision`       | `int`   | 1       | enable/disable mixed precision                                                                           |
| `--patience`              | `int`   | 0       | patience for early stopping                                                                              |
| `--calc_metrics_interval` | `int`   | 1       | interval to calculate accuracy                                                                           |
| `--slice_of_data`         | `int`   | None    | gets only a slice of data. None is get all of data                                                       |
| `--early_stopping_metric` | `str`   | None    | metric for early stopping                                                                                |
| `--init`                  | `int`   | 1       | command for enabling or disabling the intialization                                                      |
| `--cos`                   | `int`   | 1       | 1-> cos annealing LR scheduler 0-> Step Lr Scheduler (spicify gamma and step size. default : 0.99 and 1) |
| `--p`                     | `float` | 0.5     | p for data augmentation                                                                                  |
| `--siamese`               | `int`   | 1       | choose between the original siamese neural network or resnext as a backbone siamese. default 1-> siamese |

## best Recipe

- cos => 0
- epochs => 50
- learning rate => 0.0001
- weight decay => 0.0001
- lr_scheduler_on => 1 (on)
- gamma => 0.95
- step_size => 1
- mixed_precision => 1
- batch_size => 64
- patience => 0
- num_workers => 4
- calc_metrics_interval => 1
- slice_of_data => None
- siamese => 0
