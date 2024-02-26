import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork2(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, kernel_size3, initial_step):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size1, 
                               stride=1, padding=0, dilation=1, groups=1, 
                               bias=True, padding_mode='zeros', device=None, 
                               dtype=None) # input channel, kernel size
        self.pool = nn.MaxPool1d(kernel_size=kernel_size2, stride=None, padding=0, 
                                  dilation=1, return_indices=False, 
                                  ceil_mode=False)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size3, 
                               stride=1, padding=0, dilation=1, groups=1, 
                               bias=True, padding_mode='zeros', device=None, 
                               dtype=None)  # input channel, kernel size
        self.l1 = nn.Linear(initial_step, 120)       # input, hidden units
        self.l2 = nn.Linear(120, 84)                 # input, hidden units
        self.l3 = nn.Linear(84, 6)                   # input, hidden units
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)                      # flatten all dimensions except batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x