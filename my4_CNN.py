import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

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

def train(epochs,Y):
    # Define the training loop
    print("train", Y)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for i, (input, labels) in enumerate(dataloader):  
            x = input.to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()

            total += loss.item()

            if i == 0 and epoch%400==0:
                yh = yhat.tolist()
                yh = yh[0]
                yht = [round(x,4) for x in yh]
                l = loss.item()
                print(f"{epoch: 4d}, {yht},\t{l}")
            
        train_loss = total/ len(dataloader)

        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader_val):
                x = inputs.to(device)
                y = labels.to(device)

                val_outputs = model(x)
                loss = criterion(val_outputs, y)
                val_loss += loss.item()

                if i == 0 and epoch%400==0:
                    yh = val_outputs.tolist()
                    yh = yh[0]
                    yht = [round(x,4) for x in yh]
                    l = loss.item()
                    print(f"validation {y[0]}")
                    print(f"{epoch: 4d}, {yht},\t{l}")
                
        val_loss /= len(dataloader_val)

        if epoch%400==0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.4f}')

def Load_dataset(path_dataset, n_dataset):
    X_data = 'X' + n_dataset
    y_data = 'y' + n_dataset
    pathX = os.path.join('.', path_dataset, X_data)
    pathy = os.path.join('.', path_dataset, y_data)

    print("Load Data.")
    with open(pathX, 'rb') as file:
        X = pickle.load(file)

    with open(pathy, 'rb') as file:
        Y = pickle.load(file)

    print(f"X shape = {X.shape}")
    print(f"Y shape = {Y.shape}")

    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    datatensor = TensorDataset(X_train, y_train)
    dataloader = DataLoader(datatensor, batch_size = 10, shuffle = False)

    datatensor_val = TensorDataset(X_val, y_val)
    dataloader_val = DataLoader(datatensor_val, batch_size = 10, shuffle = False)

    datatensor_test = TensorDataset(X_test, y_test)
    dataloader_test = DataLoader(datatensor_test, batch_size = 10, shuffle = False)

    #for _, j in enumerate([dataloader, dataloader_val, dataloader_test]):
    #    num_batches = len(j)
    #    batch_size = j.batch_size
    #
    #    print("Numero totale di batch nel DataLoader:", num_batches)
    #    print("Dimensione di ciascun batch:", batch_size)

    return dataloader, dataloader_val, dataloader_test

def denormalize_y(y):
    y_vel = y[:,:3]
    y_ang = y[:,-3:]
    y_v_n = y_vel*0.0002
    y_a_n = y_ang*(np.pi)
    y_norm = np.column_stack([y_v_n,y_a_n])
    return y_norm

def test_accuracy(net, dataloader):
    net.eval()
    total_errors = []
    total_lengths = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = denormalize_y(labels)
            labels = torch.tensor(labels)
            labels = labels.to(device)
            outputs = net(inputs)
            errors = torch.abs(labels - outputs)
            total_errors.append(errors)
            total_lengths += inputs.size(0)

    total_errors = torch.cat(total_errors, dim=0)
    avg_errors = total_errors.mean(dim=0)

    # Calcolo delle precisioni 
    tollerance_velocity = 0.0001
    tollerance_position = 0.0174533
    accuracies_V = ((avg_errors[:3] <= tollerance_velocity).float().mean() * 100).item()
    accuracies_P = ((avg_errors[3:] <= tollerance_position).float().mean() * 100).item()

    return accuracies_V, accuracies_P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_dataset = sys.argv[1]
n_dataset = sys.argv[2]
epochs = sys.argv[3]
epochs = np.array(epochs, dtype = np.int64)

dataloader, dataloader_val, dataloader_test = Load_dataset(path_dataset, n_dataset)

model = FC()
model.to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

data_iter = iter(dataloader)
first_batch = next(data_iter)
X, Y = first_batch

train(epochs, Y[0])
print("End Train.")

# Test del modello
model.eval()
total_test_loss = 0
total_test_samples = 0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):  
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        total_test_loss += loss.item() * inputs.size(0)
        total_test_samples += inputs.size(0)

mse = total_test_loss / total_test_samples
print(f'Mean Square Error on the test set: {mse:.4f}')

# Calcolo e stampa delle precisioni per il validation set
accuracies_V, accuracies_P = test_accuracy(model, dataloader)
print("Train set:")
print(f'Velocity accuracy: {accuracies_V:.2f} %')
print(f'Position accuracy: {accuracies_P:.2f} %')

# Calcolo e stampa delle precisioni per il validation set
accuracies_V, accuracies_P = test_accuracy(model, dataloader_val)
print("Validation set:")
print(f'Velocity accuracy: {accuracies_V:.2f} %')
print(f'Position accuracy: {accuracies_P:.2f} %')

# Calcolo e stampa delle precisioni per il test set
accuracies_V, accuracies_P = test_accuracy(model, dataloader_test)
print("Test set:")
print(f'Velocity accuracy: {accuracies_V:.2f} %')
print(f'Position accuracy: {accuracies_P:.2f} %')