import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


device = (
    f"cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Carico modello

from Rete_Neurale import NeuralNetwork

net = NeuralNetwork()
net.to(device)
weights=torch.load("modello_addestrato.pth")
net.load_state_dict(weights)

################# Sint Set #################################
def load_data(input_path,label_path):
    with open(input_path, 'r') as file:
        line = file.readlines()
        line = np.float32(line)
        inputs = torch.from_numpy(line).unsqueeze(0).unsqueeze(1).float()

    with open(label_path, 'r') as file:
        line = file.readlines()
        line = np.float32(line)
        labels = torch.from_numpy(line).unsqueeze(0).float()
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    return dataloader
    
################# Sint Set #################################
dataloader = load_data("./Tesi_results/CdL_StS_Real.txt","./Tesi_results/Initial_param_StS_Real.txt")

with torch.no_grad():
    for data in dataloader:
        inputs, real = data[0].to(device), data[1].to(device)
    predict = net(inputs.to(device))
print()
print(f"valori reali StS: {real}")
print()
print(f"valori trovati con la NN: {predict}")

################# Real Set #################################
#dataloader = load_data("./Tesi_results/CdL_StR_Real.txt","./Tesi_results/Initial_param_StR_Real.txt")

#with torch.no_grad():
#    inputs, real = dataloader[0].to(device), dataloader[1].to(device)
#    predict = net(inputs.to(device))
#print()
#print(f"valori reali StR: {real}")
#print()
#print(f"valori trovati con la NN: {predict}")

