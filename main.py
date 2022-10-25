import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import os
import torch.utils.data as data
from PIL import Image



class custom_test_loader(data.Dataset):
    def __init__(self, root, transforms):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imgs = os.listdir(root)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_loc = os.path.join(self.root, self.imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

class custom_train_loader(data.Dataset):
    def __init__(self, root, transforms):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imgs = os.listdir(root)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_loc = os.path.join(self.root, self.imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        print(img_loc)
        tensor_image = self.transforms(image)
        return tensor_image




transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
trainset = custom_train_loader('../train', transforms=transform)
trainloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=32, 
                                         shuffle=True)


testset = custom_test_loader('../test1', transforms=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=32, 
                                         shuffle=True)




