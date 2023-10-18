import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork2(nn.Module):
    def __init__(self,filter_size1, kernel_size1, filter_size2, kernel_size2, kernel_size3, initial_step):
        super(NeuralNetwork2, self).__init__()
        self.conv1 = nn.Conv1d(1, filter_size1, kernel_size=kernel_size1, padding=1) # input channel, filter size, kernel size
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_size3)
        self.conv2 = nn.Conv1d(filter_size1, filter_size2,kernel_size=kernel_size2, padding=1)  # input channel, filter size, kernel size
        self.l1 = nn.Linear(initial_step, 120)              # input, hidden units
        self.l2 = nn.Linear(120, 84)                 # input, hidden units
        self.l3 = nn.Linear(84, 6)                   # input, hidden units
        
    def forward(self,x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.l1(x))
        x = nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x