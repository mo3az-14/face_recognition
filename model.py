import torch 
import torch.nn as nn 

class Model(nn.Module): 
    def __init__(self ): 
        """ 
        model input shape is (batch size , CH, W , H) 
        """
        super(Model, self).__init__()
        # backbone model
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   
        )
        self.linear = nn.Sequential(nn.Linear(256*8*8, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x =  self.linear (x)  
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out