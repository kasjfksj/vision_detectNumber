import torch.nn as nn
import torch
from torchsummary import summary
class Net(nn.Module):
    def __init__(self) :
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels = 8,kernel_size=9,padding=2,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=7,padding=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,padding=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(in_features=1568,out_features=128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=128,out_features=10),
            nn.Softmax(dim=1)


        )
    
    
    def forward(self,input):
        output = self.model(input)
        return output
