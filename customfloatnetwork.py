from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import easydict
import os
from copy import deepcopy
# Importing here so we can use in the model if needed
from clampfloat import *

device = 'cpu'
if torch.cuda.is_available() and args.cuda:
    device = 'cuda'
elif torch.backends.mps.is_available() and args.mps:
    device = 'mps'


allowInputClamp = True
allowLayerClamp = True
print('Using device: ' + str(device))

@dataclass
class CustomFloat:
    signed: bool = True
    exponent: int = 8
    mantisa: int = 23

# Function that runs the testing set on the passed model using the passed data loader
#takes in a floatFormat otherwise defaults to 32bit float. You can not use more than 8 bits of exponent or 23 bits of mantisa or 1 bit of sign
# Prints: accuracy of test set
# Returns: accuracy of test set
def test_model_float(model,test_loader, floatFormat = CustomFloat(), en_print=True):
    criterion = nn.CrossEntropyLoss()
    
    criterion=criterion.to(device)
    model = model.to(device)
    correct = 0
    total = 0
    model.eval()

    #determine what the value of the signed bit should be
    signBits = 1 if floatFormat.signed else 0

    #determine if we need to implement custom floats
    enableCustomFloats = not (floatFormat.signed and floatFormat.exponent == 8 and floatFormat.mantisa == 23)


    for images, labels in test_loader:

        #clamp the floats if we are using a custom float type
        if allowInputClamp and enableCustomFloats:
            hold = images.detach().numpy()
            vec_clamp_float(hold , signBits, floatFormat.exponent, floatFormat.mantisa)
            images = torch.from_numpy(hold)
        
        
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        testloss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    if en_print:
        print('Accuracy for test images: % d %%' % (100 * correct / total))
    accuracy = 100*correct/total
    accuracy.to('cpu')
    return accuracy.item()

# Function that trains a model on the training set with set parameters
# Prints: progress of training
# Returns: trained model

def train_model_float(model, optimizer, train_loader, data_set_len, floatFormat = CustomFloat(), num_epochs=5, clamp_bias = False , batch_size = 64):
    criterion = nn.CrossEntropyLoss()
    
    criterion=criterion.to(device)
    model = model.to(device)

    #determine what the value of the signed bit should be
    signBits = 1 if floatFormat.signed else 0

    #determine if we need to implement custom floats
    enableCustomFloats = not (floatFormat.signed and floatFormat.exponent == 8 and floatFormat.mantisa == 23)

    #clean up the default biases and weights and make sure they are in range
    if enableCustomFloats:
        with torch.no_grad():
            fixLayers(model,floatFormat,clamp_bias)


    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            #clamp the floats if we are using a custom float type
            #images = Variable(images.view(-1, 28 * 28))
            if allowInputClamp and enableCustomFloats:
                hold = images.detach().numpy()
                vec_clamp_float(hold , signBits, floatFormat.exponent, floatFormat.mantisa)
                images = torch.from_numpy(hold)
                
            images = images.to(device)
            labels = Variable(labels).to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # L1norm = model.parameters()
            # arr = []
            # for name,param in model.named_parameters():
            #     if 'weight' in name.split('.'):
            #         arr.append(param)
            # L1loss = 0
            # for Losstmp in arr:
            #     L1loss = L1loss+Losstmp.abs().mean()
            # if(args.L1norm==True):
            #     if len(arr)>0:
            #         loss = loss+L1loss/len(arr)

            loss.backward()
            optimizer.step()

            if enableCustomFloats:
                #this might need to be blocked with grad update stuff
                with torch.no_grad():
                    fixLayers(model,floatFormat,clamp_bias)



            if (i + 1) % 600 == 0:
                print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                        % (epoch + 1, num_epochs, i + 1,
                        data_set_len // batch_size, loss.data.item()))
    return model


def fixLayers(model: torch.nn.Module, cfloat:CustomFloat, fixBias = True):
    if not allowLayerClamp:
        return
    on = 1 if cfloat.signed else 0
    for name, param in model.named_parameters(): 
        if param.requires_grad:
            if("bias" in name and fixBias):
                hData = param.data.detach().numpy()
                vec_clamp_float(hData,on,cfloat.exponent,cfloat.mantisa)
                param.data = nn.Parameter(torch.from_numpy(hData))
            elif("weight" in name):
                hData = param.data.detach().numpy()
                vec_clamp_float(hData,on,cfloat.exponent,cfloat.mantisa)
                param.data = nn.Parameter(torch.from_numpy(hData))
