import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    a=input("quale gpu vuoi usare? [0/1] ")

    while a not in ['0', '1']:
        print("Input non valido. inserire '0' o '1'.")
        a = input("Quale GPU vuoi usare? [0/1]: ")

device = (
    f"cuda:{a}"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



if torch.cuda.is_available():

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 25, kernel_size=3).cuda() # input channel, filter size, kernel size
            self.pool = nn.MaxPool1d(kernel_size=2).cuda()      # kernel size, padding
            self.conv2 = nn.Conv1d(25,50,kernel_size=3).cuda()  # input channel, filter size, kernel size
            self.l1 = nn.Linear(29900, 2500).cuda()               # input, hidden units
            self.l2 = nn.Linear(2500, 25).cuda()                  # input, hidden units
            self.l3 = nn.Linear(25, 6).cuda()                   # input, hidden units
        
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
            self.pool = nn.MaxPool1d(kernel_size=2)      # kernel size, padding
            self.conv2 = nn.Conv1d(25,50,kernel_size=3)  # input channel, filter size, kernel size
            self.l1 = nn.Linear(29900, 2500)             # input, hidden units
            self.l2 = nn.Linear(2500, 25)                # input, hidden units
            self.l3 = nn.Linear(25, 6)                   # input, hidden units
        
        def forward(self,x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.l3(x)
            return x
    
    net = NeuralNetwork()
    

# iperparametri
lr = 0.2       # learning rate
momentum = 0.001 # momentum
max_epoch = 30       # numero di epoche
batch_size = 20  # batch size
scaler = GradScaler()

# ottimizzatori
if torch.cuda.is_available():
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()
#optimizer = optim.Adam(net.parameters(), lr)
optimizer = optim.SGD(net.parameters(), lr)


# Download e verifica database
# https://drive.google.com/drive/folders/1ampEXA5FIC4oYK3hpxx0yiYZLufgfoi9?usp=sharing
# second try
import gdown
import os
url = 'https://drive.google.com/file/d/1sX_LEnDldcNI-Zy9H6L2pvtGvNgDSvs-/view?usp=drive_link'
output = 'X2.npy'

# Verifica se il file è già stato scaricato
if not os.path.isfile(output):
    # Se il file non esiste, esegui il download
    gdown.download(url, output, quiet=False)

# Ora verifica se il file è stato scaricato correttamente
if os.path.isfile(output) and os.path.getsize(output) > 0:
    print(f'Il file {output} è stato scaricato correttamente.')
else:
    print(f'Il file {output} non è stato scaricato correttamente o è vuoto.')


import gdown
url = 'https://drive.google.com/file/d/1D6I0kxObp61DGlIpqg7axsn9KEHlGubj/view?usp=drive_link'
output = 'y2.npy'

# Verifica se il file è già stato scaricato
if not os.path.isfile(output):
    # Se il file non esiste, esegui il download
    gdown.download(url, output, quiet=False)

# Ora verifica se il file è stato scaricato correttamente
if os.path.isfile(output) and os.path.getsize(output) > 0:
    print(f'Il file {output} è stato scaricato correttamente.')
else:
    print(f'Il file {output} non è stato scaricato correttamente o è vuoto.')



# carico dataset
with open('X2', 'rb') as file:
    X = pickle.load(file)

with open('y2', 'rb') as file:
    y = pickle.load(file)

# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Variabile per controllare se eseguire l'addestramento o meno
train_model = input('Eseguire addestramento? [Yes/No] ').lower()

while train_model not in ['y', 'n', 'yes', 'no']:
    print("Input non valido. inserire 'y', 'n', 'yes' o 'no'.")
    a = input("Eseguire addestramento ridotto? [Yes/No] ").lower()

if train_model == 'y' or train_model == 'yes':
    train_model = True
elif train_model == 'n' or train_model == 'no':
    train_model = False


if train_model:
# Riduzione del dataset
    print('Il dataset contiene 100.000 samples')
    reduce = input('Eseguire addestramento ridotto? [Yes/No] ').lower()

    while reduce not in ['y', 'n', 'yes', 'no']:
        print("Input non valido. inserire 'Y', 'N', 'Yes' o 'No'.")
        a = input("Eseguire addestramento ridotto? [Yes/No] ").lower()

    if reduce == 'y' or reduce == 'yes':
        reduce = True
    elif reduce == 'n' or reduce == 'no':
        reduce = False
else: 
    reduce = False

if reduce:
    r_input=input("Inserire percentuale di riduzione del dataset: [1:99] ")
    try:
        r = float(r_input)
        r2 = 100-r
        r = r/100
        print(f'Dataset utilizzato: {r2}%')
    except ValueError:
        print('Input non valido. Inserire un numero valido in formato float.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=seed)
    inputs = torch.from_numpy(X_train).unsqueeze(1).float()
    labels = torch.from_numpy(y_train).float()
else:
    inputs = torch.from_numpy(X).unsqueeze(1).float()
    labels = torch.from_numpy(y).float()

train_dataset = TensorDataset(inputs, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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

#os.remove(Xtemp_file_path)
#os.remove(ytemp_file_path)
