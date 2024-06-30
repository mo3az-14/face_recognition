import torch
import numpy as np 

rng = np.random.default_rng()

from torchvision import datasets

class Pair_Data_Loader(datasets.ImageFolder) :
    def __init__(self,indices=None, *arg, **kw):
        super( Pair_Data_Loader, self).__init__(*arg,**kw)
        if indices is not None:
            self.samples= [self.samples[i] for i in indices]
            self.targets= [self.targets[i] for i in indices]
        self.n_pairs= len(self.samples)
        self.train_pairs= self.gen_example()

    def __len__(self): 
        return (len(self.targets))

    def gen_example(self): 
        labels= torch.Tensor(self.targets) 
        positive = [] 
        negative = []
        for x in range(len(labels)): 
            idx = x
            idx_matches = np.where(labels.numpy() == labels[idx].numpy())[0]
            idx_no_matches = np.where(labels.numpy()!= labels[idx].numpy())[0]
            idx_1 = np.random.choice(idx_matches , 2, replace = True )
            idx_0 = np.random.choice(idx_no_matches , 1 , replace = True )
            positive.append([idx_1[0] , idx_1[1], 1  ])
            negative.append([idx_0[0] , idx_1[0], 0 ])
        result = np.vstack((positive, negative ))
        rng.shuffle(result)
        return result 
                        
    def set_pairs(self , pairs ): 
        self.train_pairs = pairs
    
    def __getitem__(self, idx):
        
        t = self.train_pairs[idx]
        
        pth1 ,_= self.samples[t[0]]
        pth2 ,_= self.samples[t[1]]
        
        img1 = self.loader(pth1)
        img2 = self.loader(pth2)

        if self.transforms is not None :
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        target = torch.tensor(t[2] , dtype = torch.float32 )
        
        return img1, img2, target