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
            #x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1) # flatten all dimensions except batch
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

    z=len(X)
    print(f'Il dataset contiene {z} samples')
    
    r_input=input("Inserire la quota parte del trainset: [1:99] ")
    if not r_input:
        r=0.25
    else:
        try:
            r2 = float(r_input)
            r = 100-r2
            r = r/100
            print(f'Trainset utilizzato: {r}%')
        except ValueError:
            print('Input non valido. Inserire un numero valido in formato float.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=seed)
    z2=len(X_train)
    z3=len(X_test)
    print(f'Il trainset contiene {z2} samples')
    print(f'Il testset contiene {z3} samples')
    inputs = torch.from_numpy(X_train).unsqueeze(1).float()
    labels = torch.from_numpy(y_train).float()
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    inputs = torch.from_numpy(X_train).unsqueeze(1).float()
    labels = torch.from_numpy(y_train).float()

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
    model_save_path = './modello_addestrato.pth'
    torch.save(net.state_dict(), model_save_path)
else:
    model_save_path = './modello_addestrato.pth'

###########################################################

# Carico modello
net=NeuralNetwork()
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Test set
inputs = torch.from_numpy(X_test).unsqueeze(1).float()
labels = torch.from_numpy(y_test).float()

test_dataset = TensorDataset(inputs, labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

dataiter = iter(test_dataloader)
#inputs, labels = next(dataiter)

# Test 
with torch.no_grad():
    predicted = net(inputs.to(device))
    reals=[]
    for data in test_dataloader:
        inputs, real = data[0].to(device), data[1].to(device)
        reals.append(real)

reals = torch.cat(reals, dim=0)

# get the accuracy for all value
errors= reals - predicted
errors= torch.Tensor.cpu(errors)

tollerance_velocity=0.01
tollerance_position=1

# error like True or False
num_row, num_col = errors.size() 
errors_V = errors[:,0:3]
errors_P = errors[:,3:6]
boolean_eV = errors_V <= tollerance_velocity
boolean_eP = errors_P <= tollerance_position

float_tensor_V = boolean_eV.float()
float_tensor_P = boolean_eP.float()


accuracies_V = float_tensor_V.mean(dim=0)
accuracies_P = float_tensor_P.mean(dim=0)

# Print accuracies
print()
accuracies_V=torch.Tensor.numpy(accuracies_V)
accuracies_P=torch.Tensor.numpy(accuracies_P)
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f}%')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f}%')

print()



