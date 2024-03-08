import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_dataset = sys.argv[1]
n_dataset = sys.argv[2]
epochs = sys.argv[3]
epochs = np.array(epochs, dtype = np.int64)
X_data = 'X' + n_dataset
y_data = 'y' + n_dataset
pathX = os.path.join('.', path_dataset, X_data)
pathy = os.path.join('.', path_dataset, y_data)

print("Load Data.")
with open(pathX, 'rb') as file:
    X = pickle.load(file)

with open(pathy, 'rb') as file:
    Y = pickle.load(file)

X=torch.tensor(X).float()
Y=torch.tensor(Y).float()
datatensor = TensorDataset(X, Y)
dataloader = DataLoader(datatensor, batch_size = 10, shuffle = False)

print(f"X shape = {X.shape}")
print(f"Y shape = {Y.shape}")


class FC(nn.Module):
    def __init__(self, hidden_neurons=2000):
        super(FC, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.stacked = nn.Sequential(
            nn.Linear(600, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 6),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.stacked(x)

model = FC()
model.to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

def train():
    # Define the training loop
    print("0: ",Y[0])
    for epoch in range(epochs):
        total = 0
        for i, (input, labels) in enumerate(dataloader):  
            x = input.to(device)
            y = labels.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)
            if i == 0 and epoch%100==0:
                yh = yhat.tolist()
                yh = yh[0]
                yht = [round(x,4) for x in yh]
                l = loss.item()
                print(f"{epoch: 4d}, {yht},\t{l}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()

train()
print("End Train.")
#print("Check Result:")
#
#for i,x in enumerate(X):
#    x = x.to(device)
#    y=Y[i].to(device)
#    yhat = model(x)
#    loss = criterion(yhat,y)
#    yht = [round(x,4) for x in yhat.tolist()]
#    l = loss.item()
#    print(f"{i}, {yht},\t{l}")
