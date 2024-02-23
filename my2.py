
import pickle
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Load Data.")
with open("./dataCNN/X8", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y8", 'rb') as file:
    Y = pickle.load(file)
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
print(f"X shape = {X.shape}")
print(f"Y shape = {Y.shape}")




class FC(nn.Module):
    def __init__(self, hidden_neurons = 2000):
        super(FC, self).__init__()
        self.stacked = nn.Sequential(
                nn.Linear(1260, hidden_neurons),
                nn.ReLU(),
                nn.Linear(hidden_neurons, 7),
                nn.Tanh()
        )

    def forward(self, x):
        return self.stacked(x)

model = FC()
model.to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

def train():
    # Define the training loop
    epochs=10000
    print("0: ",Y[0])
    for epoch in range(epochs):
        total = 0
        for i, (x, y) in enumerate(zip(X, Y)):
            x = x.to(device) 
            y = y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)
            if i == 0 and epoch%100==0:
                yh = yhat.tolist()
                yht = [round(x,4) for x in yh]
                l = loss.item()
                print(f"{epoch: 4d}, {yht},\t{l}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()

train()
print("End Train.")
print("Check Result:")

for i,x in enumerate(X):
    x = x.to(device)
    y=Y[i].to(device)
    yhat = model(x)
    loss = criterion(yhat,y)
    yht = [round(x,4) for x in yhat.tolist()]
    l = loss.item()
    print(f"{i}, {yht},\t{l}")
