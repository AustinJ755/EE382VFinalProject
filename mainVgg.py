import time

from torchvision import datasets
import torch.nn.functional as F
#from torchsummary import summary
#from copy import deepcopy
from customfloatnetwork import *
import easydict
from torch.utils.data import random_split
import numpy as np
#from matplotlib.ticker import LinearLocator
#from matplotlib import cm
#import matplotlib.pyplot as plt
#from torch.autograd import Variable
#import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from dataclasses import dataclass
import torch
from torchvision.io import read_image
from torchvision.models import AlexNet
 
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)
# from lib import *
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
floatinfo = CustomFloat(True,8,23)
BATCH_SIZE=64
num_epochs=5
lr=1e-4
class_size=10

tranform_train = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
tranform_test = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#prep the train, validation and test dataset
torch.manual_seed(2021)
train = dsets.CIFAR10("data/", train=True, download=True, transform=tranform_train) 
val_size = 10000 
train_size = len(train) - val_size
train, val = random_split(train, [train_size, val_size]) 
test = dsets.CIFAR10("data/", train=False, download=True, transform=tranform_test) 

#  train, val and test datasets to the dataloader
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)



class VGG16_NET(nn.Module):
    def __init__(self):
        super(VGG16_NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x
    
model = VGG16_NET() 
model = model.to(device=device) 

load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr)
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
f = open("VGGTest"+timestr+".txt", "a")
f.write("Starting Test\n")
# for epoch in range(num_epochs): #I decided to train the model for 50 epochs
#     loss_var = 0
    
#     for idx, (images, labels) in enumerate(train_loader):
#         images = images.to(device=device)
#         labels = labels.to(device=device)
#         ## Forward Pass
#         optimizer.zero_grad()
#         scores = model(images)
#         loss = criterion(scores,labels)
#         loss.backward()
#         optimizer.step()
#         loss_var += loss.item()
#         if idx%64==0:
#             print(f'Epoch [{epoch+1}/{num_epochs}] || Step [{idx+1}/{len(train_loader)}] || Loss:{loss_var/len(train_loader)}')
#     print(f"Loss at epoch {epoch+1} || {loss_var/len(train_loader)}")

#     with torch.no_grad():
#         correct = 0
#         samples = 0
#         for idx, (images, labels) in enumerate(val_loader):
#             images = images.to(device=device)
#             labels = labels.to(device=device)
#             outputs = model(images)
#             _, preds = outputs.max(1)
#             correct += (preds == labels).sum()
#             samples += preds.size(0)
#         print(f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")

#Get Data for purely trained test
startt =time.time()
par_acc = []
for i in range(2,9,2):
    for j in range(1,24,2):
        print(i,",",j,"- Training")
        model = VGG16_NET() 
        model = model.to(device=device) 
        floatinfo = CustomFloat(True,i,j)
        optimizer = torch.optim.Adam(model.parameters(), lr= lr) 
        train_model_float(device,model,optimizer,train_loader,len(train),floatinfo,num_epochs,True,BATCH_SIZE)
        model.eval()
        print(i,",",j,"- Testing")
        accuracy = test_model_float(device,model,val_loader,floatinfo)
        par_acc.append(accuracy)
        print(i,",",j,"- Accuracy: ",accuracy)
        f.write('{0},{1} - {2}\n'.format(i,j,accuracy))

f.close()
print(par_acc)
endt =time.time()
elapsed_time = endt - startt
print('Execution time:', elapsed_time, 'seconds')

