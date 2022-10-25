#pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, datasets
#other imports
import os
from PIL import Image
from time import time

#use TEMP for values not calculated yet.
TEMP = 100

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
        file_split = img_loc.split('.')
        label = file_split[0]
        if label.lower() == 'dog':
            label = 0
        else:
            label = 1
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transforms(image)
        tensor_label = torch.tensor(label) 
        return tensor_image.to('cuda'), tensor_label.to('cuda')




transform = transforms.Compose([transforms.Resize((100,100)),
                                transforms.ToTensor()])
trainset = custom_train_loader('../train', transforms=transform)
trainloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=1000, 
                                         shuffle=True, num_workers=4)


testset = custom_test_loader('../test1', transforms=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=1000, 
                                         shuffle=True, num_workers=4)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 100, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(9 * 9 * 100, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 100 * 100 * 3 -> 96 * 96 * 6 -> 48 * 48 * 6
        x = self.pool(F.relu(self.conv2(x))) # 48 * 48 * 6 -> 44 * 44 * 16 -> 22 * 22 * 16
        x = self.pool(F.relu(self.conv3(x))) # 22 * 22 * 16 -> 18 * 18 * 100 -> 9 * 9 * 100
        x = x.view(-1, 9 * 9 * 100) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

if __name__ == "__main__":
    model = CNN().to('cuda')
    epochs = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01)



    def training_loop():
        for epoch in range(epochs):
            print('epoch')
            start = time()
            model.train() 
            print(len(trainloader))
            for i, data in enumerate(trainloader):
                img, label = data
                if i % 100 == 0:
                    print(i)
                out = model(img)
                loss = loss_fn(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            end = time()
            print(f'epoch {epoch} ended in {end - start} seconds')

        print('training done!')

        torch.save(model.state_dict(), './Model.pth')
            


    training_loop()
