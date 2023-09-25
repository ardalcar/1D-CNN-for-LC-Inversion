import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 25, kernel_size=3) # input channel, filter size, kernel size
        self.pool = nn.MaxPool1d(kernel_size=2)      # kernel size, padding
        self.conv2 = nn.Conv1d(25,50,kernel_size=3)  # input channel, filter size, kernel size
        self.l1 = nn.Linear(29900, 120)              # input, hidden units
        self.l2 = nn.Linear(120, 84)                 # input, hidden units
        self.l3 = nn.Linear(84, 6)                   # input, hidden units
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x