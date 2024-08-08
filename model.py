import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


class original_siamese(nn.Module):
    def __init__(self):
        """
        model input shape is (batch size , CH, W , H)
        """
        super(original_siamese, self).__init__()
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
        self.linear = nn.Sequential(nn.Linear(256 * 8 * 8, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out


class resnext_50(nn.Module):
    def __init__(self):
        super(resnext_50, self).__init__()
        self.backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        input_dim = self.backbone.fc.in_features
        for i in self.backbone.parameters():
            i.requires_grad = False
        self.backbone.fc = nn.Identity()
        self.linear1 = nn.Sequential(nn.Linear(input_dim, 1024), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(1024, 512), nn.Linear(512, 1))

    def pre_forward(self, img):
        x = self.backbone(img)
        x = self.linear1(x)
        return x

    def forward(self, img1, img2):
        img1 = self.pre_forward(img1)
        img2 = self.pre_forward(img2)
        dist = torch.abs(img1 - img2)
        out = self.out(dist)

        return out
