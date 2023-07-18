import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import Dataset
import requests, os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


#class CustomDataset(Dataset):
#    def __init__(self, X_file, y_file):
#        self.X_data = pickle.load(X_file)
#        self.y_data = pickle.load(y_file)

#    def __len__(self):
#        return len(self.X_data)

#    def __getitem__(self, index):
#        X_sample = torch.from_numpy(self.X_data[index]).unsqueeze(0).float()
#        y_sample = torch.from_numpy(self.y_data[index]).float()
#        return X_sample, y_sample

#X_file = 'X.pickle'
#y_file = 'y.pickle'
#dataset = CustomDataset(X_file, y_file)

with open('X.pickle', 'rb') as file:
    X = pickle.load(file)

with open('y.pickle', 'rb') as file:
    y = pickle.load(file)

# Database da file
#X = np.load('X.npy')
#y = np.load('y.npy')

# Divisione del dataset in addestramento e verifica in modo casuale
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

# Conversione dei dati di input in tensori di PyTorch
#inputs = torch.from_numpy(X_train).unsqueeze(1).float()
#labels = torch.from_numpy(y_train).float()

if torch.cuda.is_available():

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork,self).__init__()
            self.conv1 = nn.Conv1d(1, 25, kernel_size=3).cuda() # input channel, filter size, kernel size
            self.pool = nn.MaxPool1d(kernel_size=2).cuda()      # kernel size, padding
            self.conv2 = nn.Conv1d(25,50,kernel_size=3).cuda()  # input channel, filter size, kernel size
            self.l1 = nn.Linear(29900, 25).cuda()               # input, hidden units
            self.l2 = nn.Linear(25, 10).cuda()                  # input, hidden units
            self.l3 = nn.Linear(10, 6).cuda()                   # input, hidden units
        
        def forward(self,x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.l3(x)
            return x
    
    net = NeuralNetwork()
    net.cuda()

else:
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork,self).__init__()
            self.conv1 = nn.Conv1d(1, 25, kernel_size=3) # input channel, filter size, kernel size
            self.pool = nn.MaxPool1d(kernel_size=2)       # kernel size, padding
            self.conv2 = nn.Conv1d(25,50,kernel_size=3)    # input channel, filter size, kernel size
            self.l1 = nn.Linear(29900, 10000)       # input, hidden units
            self.l2 = nn.Linear(10000, 1000)       # input, hidden units
            self.l3 = nn.Linear(1000, 500)       # input, hidden units
            self.l4 = nn.Linear(500, 100)       # input, hidden units
            self.l5 = nn.Linear(100, 25)       # input, hidden units
            self.l6 = nn.Linear(25, 10)        # input, hidden units
            self.l7 = nn.Linear(10, 6)          # input, hidden units
        
        def forward(self,x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            x = F.relu(self.l4(x))
            x = F.relu(self.l5(x))
            x = F.relu(self.l6(x))
            x = F.relu(self.l7(x))
            x = self.l3(x)
            return x
    
    net = NeuralNetwork()
    

# iperparametri
lr = 0.2       # learning rate
momentum = 0.001 # momentum
max_epoch = 10       # numero di epoche
batch_size = 5  # batch size
scaler = GradScaler()

# ottimizzatori
if torch.cuda.is_available():
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()
#optimizer = optim.Adam(net.parameters(), lr)
optimizer = optim.SGD(net.parameters(), lr)

# Definizione dataloader per caricare i dati di addestramento
#Xurl = 'https://drive.google.com/file/d/11Tn7I1_hWol4h8ku4YWA_tx3om41VNnu/view?usp=drive_link'
#yurl = 'https://drive.google.com/file/d/1OcsGbxL562CaN9SDZHvH_pfOnMhOlPuQ/view?usp=drive_link'

# Scarica il contenuto del file
#Xresponse = requests.get(Xurl)
#yresponse = requests.get(yurl)

# Salvare i dati in un file temporaneo
#Xtemp_file_path = 'Xtemp.npy'
#with open(Xtemp_file_path, 'wb') as temp_file:
#    temp_file.write(Xresponse.content)

#ytemp_file_path = 'ytemp.npy'
#with open(Xtemp_file_path, 'wb') as temp_file:
#    temp_file.write(yresponse.content)


# Carica i dati dal file temporaneo come array numpy
#inputs = np.load(Xtemp_file_path, allow_pickle=True)
#labels = np.load(ytemp_file_path)


#train_dataset = torch.utils.data.TensorDataset(inputs, labels)
train_dataset = torch.utils.data.TensorDataset(X, y)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Variabile per controllare se eseguire l'addestramento o meno
#train_model = False
train_model = True
        
# Ciclo di addestramento
if train_model:
    for epoch in range(max_epoch):
        net.train()
        total_loss = 0

        for batch in train_dataloader:
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            
                
            optimizer.zero_grad()

            with autocast():
                outputs = net(batch_inputs)
                loss = criterion(outputs, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{max_epoch}], Loss: {avg_loss}")

    # Salva il modello addestrato
    torch.save(net.state_dict(), "modello_addestrato.pth")

# Elimina il file temporaneo

os.remove(Xtemp_file_path)
os.remove(ytemp_file_path)
