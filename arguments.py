import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="parameteres for training")

    parser.add_argument('--epochs', type=bool , default=5 , help ='epochs. default 5')
    parser.add_argument('--batch_size', type=int, default=128, help ='batch size. default 128')
    parser.add_argument('--num_worksers', type = int , default=2 , help='number of workers for data loader. default 2')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the model. default 0.0001')
    parser.add_argument('--lr_scheduler', type=bool, default=True, help='Learning rate for the model. default True')
    parser.add_argument('--lr_gamma', type=float, default=0.999, help='gamma for the learning rate scheduler. default 0.999')
    parser.add_argument('--lr_step_size', type=int, default=1, help='step size for learning rate scheduler. default 1')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay. default 0')
    parser.add_argument('--mixed_precision', type=bool, default=True , help ='enable mixed precision. default True')
    parser.add_argument('--early_stopping', type=bool, default=True , help ='enable early_stopping. default True')
    parser.add_argument('--patience', type=int, default=20, help ='patience for early stopping. default 20')
 
    args = parser.parse_args()
    return args
    

    
    