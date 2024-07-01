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
from torch.optim.lr_scheduler import StepLR
import config
from data_loaders import Pair_Data_Loader
from model import Model
import copy 
from arguments import get_arguments

# training
def train_step(model: torch.nn.Module,
               train_data:torch.utils.data.DataLoader ,
               loss_fn:torch.nn,
               optimizer:torch.optim,
               mixed_precision_on : bool = True,
               device: str = "cuda"):
    
    model.train()    
    train_loss:float = 0.0
    
    for  (first , second , target ) in tqdm.tqdm(train_data):
        
        model.train()       
        
        for param in model.parameters():
            param.grad=None
        
        first , second , target = first.to(device , non_blocking=  True) , second.to(device , non_blocking=  True) , target.to(device , non_blocking=  True)

        if  mixed_precision_on : 
            
            scaler = amp.GradScaler()
            with amp.autocast():
                output = model(first , second).squeeze() 
                loss = loss_fn (output , target)
            
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            
            scaler.update()
        else: 
            output = model(first , second).squeeze() 
            
            loss = loss_fn (output , target)
            
            loss.backward()
            
            optimizer.step()

        train_loss+= loss.item()
          
    return train_loss
    
# testing
def valid_step(model: torch.nn.Module,
               test_data:torch.utils.data.DataLoader ,
               loss_fn : torch.nn,
               device: str, 
               mixed_precision_on:bool = True):
    
    test_loss = 0.0
    
    with torch.no_grad() : 
        model.eval()
        for (first , second , target ) in tqdm.tqdm(test_data):
            
            first, second, target = first.to(device, non_blocking= True), second.to(device, non_blocking=True),\
                target.to(device , non_blocking=  True)
            if mixed_precision_on :
                # saves memory
                with amp.autocast():                
                    output = model(first , second).squeeze()
                    loss = loss_fn (output , target )
            else : 
                output = model(first , second).squeeze()
                loss = loss_fn (output , target )  
            
            test_loss += loss.item() 

    return test_loss

# training loop 
def train_loop(model: torch.nn.Module,
               train_data:torch.utils.data.DataLoader ,
               test_data:torch.utils.data.DataLoader ,
               loss_fn:torch.nn,
               optimizer:torch.optim ,
               device: str = "cuda" ,
               epochs : int = 5,
               early_stopping:bool = False,
               patience:int = 20 , 
               lr_scheduler:torch.optim.lr_scheduler = None,
               mixed_precision_on: bool = True ):
    train_loss_acc = []
    test_loss_acc= []
    
    best_loss= float('inf')
    epochs_without_imporvement = 0 
        
    for i in range(epochs):    
        
        # triaing step
        train_loss = train_step(model, train_data, loss_fn = loss_fn, optimizer = optimizer ,
                                device = device, mixed_precision_on = mixed_precision_on)
        # validation step 
        test_loss = valid_step(model , test_data , loss_fn , device)

        # early stopping
        if early_stopping:
            if train_loss < best_loss : 
                best_loss = train_loss 
                epochs_without_imporvement = 0 
                best_model_wts = copy.deepcopy(model.state_dict())
            else: 
                epochs_without_imporvement +=1

            if epochs_without_imporvement >= patience :
                print ("early stopping activated ")
                break
        
        # learning rate scheduler 
        if lr_scheduler is not None:
            print ('learning rate update')    
            lr_scheduler.step()
        
        train_loss_acc.append(train_loss)
        test_loss_acc.append(test_loss)
        print(f"train loss: {train_loss:.4f} test loss: {test_loss:.4f}@ epoch {i}")

    model.load_state_dict(best_model_wts)
    return train_loss_acc , test_loss_acc

# intializing weights
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0) 

if __name__ == '__main__':

    # get arguments from the console. defaults are in arguments.py  
    args = get_arguments()
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    lr_gamma = args.lr_gamma
    lr_step_size = args.lr_step_size
    lr_scheduler = args.lr_scheduler
    mixed_precision = args.mixed_precision
    batch_size = args.batch_size
    early_stopping = args.early_stopping
    patience = args.patience
    epochs = args.epochs
    
    torch.backends.cudnn.benchmark = True
    
    rng = np.random.default_rng()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model().to(device)
        
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate , weight_decay= weight_decay)

    scheduler = StepLR(optimizer, step_size= lr_step_size , gamma = lr_gamma) if lr_scheduler else  None

    data_transform= data_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True, ),
            transforms.Resize(size=config.IMAGE_SIZE ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    pair_data_train = Pair_Data_Loader( root=config.TRAIN_DATASET , transform = data_transform )
    pair_dataloader_train  = DataLoader(pair_data_train, batch_size=batch_size, shuffle=False, num_workers=2 , pin_memory=True)

    pair_data_test = Pair_Data_Loader( root=config.TEST_DATASET, transform=data_transform )
    pair_dataloader_test = DataLoader(pair_data_test, batch_size=batch_size, shuffle=False, num_workers=2 , pin_memory=True )
    
    loss_function = nn.BCEWithLogitsLoss(reduction = "mean" )
    
    model.apply(initialize_weights)

    train_loss , test_loss = train_loop(model, pair_dataloader_train , pair_dataloader_test, 
                                        loss_function, optimizer, early_stopping= early_stopping, patience = patience,  
                                        device = device, epochs= epochs , lr_scheduler = scheduler )
    
    print(f'Final train loss: {train_loss} , Final test loss: {test_loss}')