import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# Neural Network 
class FC(nn.Module):
    def __init__(self, intermediate_size1, intermediate_size2):
        super(FC, self).__init__()

        self.fc1 = nn.Linear(1260, intermediate_size1)
        self.fc2 = nn.Linear(intermediate_size1, intermediate_size2)
        self.fc3 = nn.Linear(intermediate_size2, 6)

        ## Inizializzazione dei pesi
        #torch.manual_seed(21)  # sempre gli stessi pesi ad ogni inizializzazione
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        nn.init.normal_(m.weight)
        #        nn.init.zeros_(m.bias)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

# Learning cicle
def learning(data_tensors, label_tensors, max_epoch):
    print(f'data shape: {data_tensors.shape}')
    print(f'label shape: {label_tensors.shape}')
    for epoch in range(max_epoch):
        net.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(zip(data_tensors, label_tensors)):

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            if i == 0 and epoch%100==0:
                yh = net.tolist()
                yht = [round(x,4) for x in yh]
                l = loss.item()
                print(f"{epoch: 4d}, {yht},\t{l}")

            if i == 10:
                specific_output = outputs  # Decimo elemento 
                specific_label = labels

                specific_output = specific_output.detach().cpu().numpy()
                specific_label = specific_label.detach().cpu().numpy()

                for j in range(specific_output.shape[0]):
                    writer.add_scalars(f'Training/Feature_{j}',
                                        {'Predicted': specific_output[j],
                                        'Actual':    specific_label[j]}, epoch)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(data_tensors)


    writer.close()
    model_save_path = 'FUCO2.pth'
    torch.save(net.state_dict(), model_save_path)

 
def MyDataLoader(X, y):
    data_tensors = torch.tensor(X, dtype=torch.float32).to(device)
    label_tensors = torch.tensor(y, dtype=torch.float32).to(device)

    return  data_tensors, label_tensors



################################### main ###################################

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

with open("./dataCNN/X8", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y8", 'rb') as file:
    y = pickle.load(file)


# Iperparametri
lr = 0.001
max_epoch = 10000
intermediate_size1=500
intermediate_size2=50
net = FC(intermediate_size1, intermediate_size2).to(device)
print(net)

model_save_path = 'FUCO2.pth'
#net.load_state_dict(torch.load(model_save_path))

# Definizione di loss function e optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0001)

# Inizializzazione di TensorBoard
writer = SummaryWriter('tensorboard/FUCO2')

data_tensors, label_tensors = MyDataLoader(X, y)
learning(data_tensors, label_tensors, max_epoch)