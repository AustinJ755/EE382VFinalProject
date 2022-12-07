import time
from torchvision import datasets
from torchsummary import summary
from copy import deepcopy
from customfloatnetwork import *
import easydict
import numpy as np
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
from dataclasses import dataclass
import torch
from torchvision.io import read_image
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)
# from lib import *

# Function for clamping weights of a given layer to a given floating point format

# img = read_image('C:\\Users\\aj755\\OneDrive - The University of Texas at Austin\\Documents\\! School Semester\\1Fall 2022\\ECE 382V\\Final Project\\crosslayer_project\\imagenetv2-top-images-format-val\\2\\807b3688188c55a32c9166a58b65a5143e69480b.jpeg')
# from torchvision.models import resnet50, ResNet50_Weights
# # Step 1: Initialize model with the best available weights
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.eval()

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)

# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")
def clamp_weights(layer, sb, eb, mb):
    np_weights = layer.weight.detach().numpy()
    # WARNING: this function is *super* slow

    np_weights = vec_clamp_float(np_weights, sb, eb, mb)

    with torch.no_grad():
        layer.weight = nn.Parameter(torch.from_numpy(np_weights))

# Hook function that can be applied by attaching to the various model layers


def clamp_weights_hook(self, input, output):
    s_bits = clamp_weights_hook.s_bits
    e_bits = clamp_weights_hook.e_bits
    m_bits = clamp_weights_hook.m_bits
    # First, detach and convert the weights to a NumPy array

    np_weights = self.weight.detach().numpy()
   # old = deepcopy(np_weights)

    # Call the vectorized clamp function
    vec_clamp_float(np_weights, s_bits, e_bits, m_bits)

    # for i in range(0,len(np_weights)):
    #     if(np_weights.item(i)!=old.item(i)):
    #         print("sbit: ",s_bits," ebits: ", e_bits, " mbits: ",m_bits)
    #         print("NEW: ",np_weights)
    #         print("OLD: ",old)
    #         exit(0)
    with torch.no_grad():
        self.weight = nn.Parameter(torch.from_numpy(np_weights))



args = easydict.EasyDict({
    "batch_size": 32,
    "epochs": 100,
    "lr": 0.001,
    "enable_cuda": True,
    "L1norm": False,
    "simpleNet": True,
    "activation": "relu",  # relu, tanh, sigmoid
    "train_curve": True,
    "optimization": "SGD",
    "cuda": False,
    "mps": False,
    "hooks": True
})
# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# MNIST Dataset (Images and Labels)
train_set = dsets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)
test_set = dsets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_set,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_set,
        batch_size = batch_size,
        shuffle = False)


class MyConvNet(nn.Module):
    def __init__(self, args):
        super(MyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin2 = nn.Linear(7*7*32, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        a1 = self.act1(b1)
        p1 = self.pool1(a1)
        c2 = self.conv2(p1)
        b2 = self.bn2(c2)
        a2 = self.act2(b2)
        p2 = self.pool2(a2)
        flt = torch.flatten(p2, start_dim=1)  # flt = p2.view(p2.size(0), -1)
        out = self.lin2(flt)
        return out


# clamp_weights_hook.s_bits = 1
# clamp_weights_hook.e_bits = floatinfo.exponent
# clamp_weights_hook.m_bits = floatinfo.mantisa
# model.conv1.register_forward_hook(clamp_weights_hook)
# #model.bn1.register_forward_hook(clamp_weights_hook)
# model.conv2.register_forward_hook(clamp_weights_hook)
# #model.bn2.register_forward_hook(clamp_weights_hook)
# model.lin2.register_forward_hook(clamp_weights_hook)
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
f = open("MyConvTest"+timestr+".txt", "w")
f.write("Starting Test\n")
epochs = 5
learn_rt = .003
# 
for i in range(1,9):
    for j in range(1,24):
        model = MyConvNet(args)
        floatinfo = CustomFloat(True,i,j)
        optimizer = torch.optim.SGD(model.parameters(), lr = learn_rt)
        print(i,",",j,"- Training")
        model = train_model_float(model,optimizer,train_loader,len(train_set),floatinfo,epochs,True,batch_size)
        model.eval()
        print(i,",",j,"- Testing")
        accuracy = test_model_float(model,test_loader,floatinfo)
        print(i,",",j,"- Accuracy: ",accuracy)
        f.write('{0},{1} - {2}\n'.format(i,j,accuracy))
        f.flush()

f.close()

        
# model = MyConvNet(args)
# model1 = MyConvNet(args)
# model2 = MyConvNet(args)
# model3 = MyConvNet(args)
# model4 = MyConvNet(args)
# model5 = MyConvNet(args)
# epochs = 8
# learn_rt = .001
# optimizer = torch.optim.SGD(model.parameters(), lr = learn_rt)
# floatinfo = CustomFloat(True,4,10)
# floatinfo1 = CustomFloat(True,6,10)
# floatinfo2 = CustomFloat(True,8,10)
# model = train_model_float(model,optimizer,train_loader,len(train_set),floatinfo,epochs,False,batch_size)
# optimizer = torch.optim.SGD(model1.parameters(), lr = learn_rt)
# model1 = train_model_float(model1,optimizer,train_loader,len(train_set),floatinfo1,epochs,False,batch_size)
# optimizer = torch.optim.SGD(model2.parameters(), lr = learn_rt)
# model2 = train_model_float(model2,optimizer,train_loader,len(train_set),floatinfo2,epochs,False,batch_size)
# optimizer = torch.optim.SGD(model3.parameters(), lr = learn_rt)
# model3 = train_model_float(model3,optimizer,train_loader,len(train_set),floatinfo,epochs,True,batch_size)
# optimizer = torch.optim.SGD(model4.parameters(), lr = learn_rt)
# model4 = train_model_float(model4,optimizer,train_loader,len(train_set),floatinfo1,epochs,True,batch_size)
# optimizer = torch.optim.SGD(model5.parameters(), lr = learn_rt)
# model5 = train_model_float(model5,optimizer,train_loader,len(train_set),floatinfo2,epochs,True,batch_size)

# print("\n---------\nM1\n")
# test_model_float(model,test_loader,floatinfo)
# print("\n---------\nM2\n")
# test_model_float(model1,test_loader,floatinfo1)
# print("\n---------\nM3\n")
# test_model_float(model2,test_loader,floatinfo2)
# print("\n---------\nM4\n")
# test_model_float(model3,test_loader,floatinfo)
# print("\n---------\nM5\n")
# test_model_float(model4,test_loader,floatinfo1)
# print("\n---------\nM6\n")
# test_model_float(model5,test_loader,floatinfo2)

# #model = MyConvNet(args)

# #train_model_float(model,)
# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# #model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)#, weights="IMAGENET1K_V2")

# from torchvision.models import resnet50, ResNet50_Weights
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.eval()
# print(weights.meta["categories"])

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# criterion = nn.CrossEntropyLoss()



# test_set = datasets.ImageFolder(root='imagenetv2-top-images-format-val', transform=preprocess)#transform=transforms.Compose([
# print(test_set.find_classes("imagenetv2-top-images-format-val"))
# #     transforms.Resize(256),
# #     transforms.CenterCrop(224),
# #     transforms.ToTensor(),
# #     normalize]))

# test_loader = torch.utils.data.DataLoader(dataset=test_set,
#                                           batch_size=1,
#                                           shuffle=False)
# correct = 0
# total = 0
# model.eval()

# # determine what the value of the signed bit should be

# # determine if we need to implement custom floats


# # for images, labels in test_loader:

# #     # clamp the floats if we are using a custom float type

# #     images = images.to(device)
# #     labels = labels.to(device)
# #     outputs = model(images)
# #     testloss = criterion(outputs, labels)
# #     _, predicted = torch.max(outputs.data, 1)
# #     total += labels.size(0)
# #     correct += (predicted == labels).sum()
# #     if total % 100:
# #         print('Accuracy for test images: % d %%' % (100 * correct / total))


# num_images = 0
# num_top1_correct = 0
# num_top5_correct = 0
# predictions = []
# #start = timer()



# #ADD CLAMPING AFTER EVERY LAYER
# enumerable = enumerate(test_loader)
# for ii, (img_input, target) in enumerable:
#     print(target)
#     result = model(img_input)
#     prediction = result.squeeze(0).softmax(0)
#     class_id = prediction.argmax().item()
#     print(class_id)
#     _, output_index = torch.topk(result,
#         k=5, dim=1, largest=True, sorted=True)
#     output_index = output_index.numpy()
#     predictions.append(output_index)
#     print(output_index[:,0])
#     for jj, correct_class in enumerate(target.numpy()):
#         if correct_class == output_index[jj, 0]:
#             num_top1_correct += 1
#         if correct_class in output_index[jj, :]:
#             num_top5_correct += 1
#     num_images += len(target)
#     if ii%10==9:

#         print('----Training Progress----')
#         print('    Evaluated {} images'.format(num_images))
#         print('    Top-1 accuracy: {:.2f}'.format(100.0 * num_top1_correct / num_images))
#         print('    Top-5 accuracy: {:.2f}'.format(100.0 * num_top5_correct / num_images))
# predictions = np.vstack(predictions)
# assert predictions.shape == (num_images, 5)
# print('----FINAL RESULTS----')
# print('    Evaluated {} images'.format(num_images))
# print('    Top-1 accuracy: {:.2f}'.format(100.0 * num_top1_correct / num_images))
# print('    Top-5 accuracy: {:.2f}'.format(100.0 * num_top5_correct / num_images))

