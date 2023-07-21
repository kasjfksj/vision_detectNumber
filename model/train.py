import torch.nn as nn
import ssl
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
import requests
import ssl
from model import Net

path = '/Users/alexanderlee/computer-vision/dataset'


transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.Resize((128,128))
])
trainData = MNIST(path,train=True,download=True,transform = transform)
testData = MNIST(path,train=False,transform = transform)
trainDataLoader =  torch.utils.data.DataLoader(dataset = trainData,batch_size = 100, shuffle = True)

testDataLoader =  torch.utils.data.DataLoader(dataset = testData,batch_size = 100, shuffle = True)


net = Net()
lossF=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = net.parameters(),lr = 1e-4)
totalLoss =0
for epoch in range(5):
    net.train(True)
    for id, (input,target) in enumerate(trainDataLoader):
        net.zero_grad()
        outputs = net(input)
        loss = lossF(outputs,target)
        predictions = torch.argmax(outputs,dim=1)
        loss.backward()
        optimizer.step()
        if id %10 ==0:
            print(epoch,": ", id, "  ", loss.item())


torch.save(net.state_dict(), "/Users/alexanderlee/computer-vision/model_parameters/model_parameter.pkl")

for id, (input,target) in enumerate(testDataLoader):

    outputs = net(input)
    predictions = torch.argmax(outputs,dim=1)
    accuracy = torch.sum(predictions==target)
print(accuracy)

