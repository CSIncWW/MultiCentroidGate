from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import SVHN

class ImageNet100():   
    def __init__(self, root, meta_root, train=True, transform=None):
        self.transform = transform
        split = "train" if train else "val" 
        metadata_path = os.path.join(meta_root, f"{split}_{100}.txt")
 
        csv = pd.read_csv(metadata_path, header=None, sep=" ") 
        self.data = csv.iloc[:, 0].map(lambda x: os.path.expanduser(os.path.join(root, "ILSVRC2012", x))).to_numpy()
        self.targets = csv.iloc[:, 1].to_numpy() 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        raise Exception()
        img = Image.open(self.data[index])
        img = self.transform(img)
        return img, self.targets[index]

class ImageNet1K():   
    def __init__(self, root, meta_root, train=True, transform=None):
        self.transform = transform

        split = "train" if train else "val" 
        metadata_path = os.path.join(meta_root, f"{split}_{1000}.txt")
 
        csv = pd.read_csv(metadata_path, header=None, sep=" ") 
        self.data = csv.iloc[:, 0].map(lambda x: os.path.expanduser(os.path.join(root, "ILSVRC2012", x))).to_numpy()
        self.targets = csv.iloc[:, 1].to_numpy()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        raise Exception()
        img = Image.open(self.data[index])
        img = self.transform(img)
        return img, self.targets[index]

class SVHNWrapper(SVHN):   
    def __init__(self, folder, split, transform, download):
        super().__init__(folder, split, transform, None, download)
        self.targets = self.labels
        self.data = np.transpose(self.data, (0, 2, 3, 1))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        raise Exception()
        img = Image.open(self.data[index])
        img = self.transform(img)
        return img, self.targets[index]

class ImageNet50():   
    def __init__(self, root, train=True, transform=None):
        self.npy = np.load(
            os.path.expanduser(
                os.path.join(root, "imgnet32", 
                ("merged_train.npy" if train else "merged_val.npy"))
            ),
            allow_pickle=True).item()
        self.data = self.npy['data'] # (b, h, w, c)
        print("len: ", self.data.shape)
        self.targets = self.npy['labels']
        self.transform = transform 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        raise Exception()
        img = Image.open(self.data[index])
        img = self.transform(img)
        return img, self.targets[index]

if __name__ == "__main__":
    k = ImageNet100("")
    import pdb; pdb.set_trace()
    print(k.data, k.targets)
    
