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


class custom_train_loader(data.Dataset):
    def __init__(self, root, transforms, train):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imgs = os.listdir(root)
        import random
        random.shuffle(self.imgs)
        if train:
            self.imgs = self.imgs[0:int(len(self.imgs) * 0.9)]
        else:
            self.imgs = self.imgs[int(len(self.imgs) * 0.9):]
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_loc = os.path.join(self.root, self.imgs[idx])
        file_split = self.imgs[idx].split('.')
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
trainset = custom_train_loader('../train', transforms=transform, train= True)
trainloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=64, 
                                         shuffle=True, num_workers=4)


testset = custom_train_loader('../train', transforms=transform, train= False)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=1, 
                                         shuffle=True, num_workers=4)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 100, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(9 * 9 * 100, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 100 * 100 * 3 -> 96 * 96 * 6 -> 48 * 48 * 6
        x = self.pool(F.relu(self.conv2(x))) # 48 * 48 * 6 -> 44 * 44 * 16 -> 22 * 22 * 16
        x = self.pool(F.relu(self.conv3(x))) # 22 * 22 * 16 -> 18 * 18 * 100 -> 9 * 9 * 100
        x = x.view(-1, 9 * 9 * 100) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))
        return x

if __name__ == "__main__":
    model = CNN().to('cuda')
    model.load_state_dict(torch.load('./Model.pth'))
    epochs = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)



    def training_loop():
        training_start = time()
        for epoch in range(epochs):
            loss = 0.0
            print(f'epoch {epoch}')
            start = time()
            model.train() 
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
            print(f'epoch {epoch} ended with loss {loss} ||| epoch {epoch} runtime : {end - start} seconds')

        training_end = time()
        print(f'training done in {int(training_end - training_start)} seconds, or {int(training_end - training_start) / 60} minutes')
        torch.save(model.state_dict(), './Model.pth')
    def testing_loop():
        model.load_state_dict(torch.load('./Model.pth'))
        correct = 0
        total = 0
        model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on test images: {100 * correct // total} %      || correct : {correct}, total : {total}')


    #training_loop()

        # torch.save(model.state_dict(), PATH)
    


    testing_loop()

