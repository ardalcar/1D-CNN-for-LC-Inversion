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
            self.pool = nn.MaxPool1d(kernel_size=2)      # kernel size, padding
            self.conv2 = nn.Conv1d(25,50,kernel_size=3)  # input channel, filter size, kernel size
            self.l1 = nn.Linear(29900, 10000)            # input, hidden units
            self.l2 = nn.Linear(10000, 1000)             # input, hidden units
            self.l3 = nn.Linear(1000, 500)               # input, hidden units
            self.l4 = nn.Linear(500, 100)                # input, hidden units
            self.l5 = nn.Linear(100, 25)                 # input, hidden units
            self.l6 = nn.Linear(25, 10)                  # input, hidden units
            self.l7 = nn.Linear(10, 6)                   # input, hidden units
        
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
