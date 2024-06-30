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

def train_step(model: torch.nn.Module,
               train_data:torch.utils.data.DataLoader ,
               loss_fn:torch.nn,
               optimizer:torch.optim,
               mixed_precision_on : bool = True,
               lr_scheduler: torch.optim.lr_scheduler = None,
               device: str = "cuda"):
    
    model.train()    
    scaler = amp.GradScaler()
    train_loss:float = 0.0
    
    for  (first , second , target ) in tqdm.tqdm(train_data):
        
        model.train()       
        
        for param in model.parameters():
            param.grad=None
        
        first , second , target = first.to(device , non_blocking=  True) , second.to(device , non_blocking=  True) , target.to(device , non_blocking=  True)

        if  mixed_precision_on : 
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
            

        if lr_scheduler is not None:
            lr_scheduler.step()            
        
        train_loss+= loss.item()
    
    return train_loss
    
    
def test_step(model: torch.nn.Module,
               test_data:torch.utils.data.DataLoader ,
               loss_fn : torch.nn,
               device: str ):
    
    test_loss = 0.0
    
    with torch.no_grad() : 
        model.eval()
        for (first , second , target ) in tqdm.tqdm(test_data):
            first, second, target = first.to(device, non_blocking= True), second.to(device, non_blocking=True),\
                target.to(device , non_blocking=  True)
            
            with amp.autocast():                
                output = model(first , second).squeeze()
                loss = loss_fn (output , target ) 
            
            test_loss += loss.item() 

    return test_loss


def train_loop(model: torch.nn.Module,
               train_data ,
               test_data,
               loss_fn,
               optimizer,
               device: torch.device = "cuda" ,
               epochs : int = 10,
               early_stopping = False,
               patience = 10):
    train_loss_acc = []
    test_loss_acc= []
    
    best_test_loss= float('inf')
    epochs_without_imporvement = 0 
        
    for i in range(epochs):    
            
        train_loss = train_step(model , train_data , loss_fn , optimizer , device )
        test_loss = test_step(model , test_data , loss_fn , optimizer , device)

        if early_stopping:

            if test_loss < best_test_loss : 
                best_test_loss = test_loss
                epochs_without_imporvement = 0 
                # best_model_wts = copy.deepcopy(model.state_dict())
            else: 
                epochs_without_imporvement +=1

            if epochs_without_imporvement >= patience :
                print ("early stopping activated ")
                break

             
        print(f"train loss: {train_loss:.4f} test loss: {test_loss:.4f}@ epoch {i}")

    # model.load_state_dict(best_model_wts)
    return train_loss_acc , test_loss_acc

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0) 

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model().to(device)
        
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001 , weight_decay= 0.01)

    scheduler = StepLR(optimizer, step_size= 1 , gamma = 0.99)

    data_transform= data_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True, ),
            transforms.Resize(size=config.IMAGE_SIZE ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    pair_data_train =Pair_Data_Loader( root=config.TRAIN_DATASET , transform = data_transform )
    
    pair_dataloader_train  = DataLoader(pair_data_train, batch_size=128, shuffle=False, num_workers=2 , pin_memory=True)

    pair_data_test =Pair_Data_Loader( root=config.TEST_DATASET, transform=data_transform )
    pair_dataloader_test = DataLoader(pair_data_test, batch_size=128, shuffle=False, num_workers=2 , pin_memory=True   )
    
    loss_function = nn.BCEWithLogitsLoss(reduction = "sum" )
    
    model.apply(initialize_weights)

    train_loss , test_loss = train_loop(model, pair_dataloader_train , pair_dataloader_test, loss_function, optimizer, early_stopping= True, patience = 5,  device = device, epochs= 5 , )
    
    print(f'Final train loss: {train_loss} , Final test loss: {test_loss}')