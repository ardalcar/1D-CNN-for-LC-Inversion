import torch
import io
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

device = (
    f"cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

################################## Neural Network ################################

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        
        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out[:, -1, :])

        return out



net = RNN(input_size=2400, 
          hidden_size=100, 
          num_layers=20, 
          output_size=6)
net.to(device)

# iperparametri
lr = 0.2          # learning rate
momentum = 0.001  # momentum
max_epoch = 500   # numero di epoche
batch_size = 20   # batch size
scaler = GradScaler()

# ottimizzatori
if torch.cuda.is_available():
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()
#optimizer = optim.Adam(net.parameters(), lr)
optimizer = optim.SGD(net.parameters(), lr)


##################################### carico dataset ##########################

with open("./dataCNN/X2", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y2", 'rb') as file:
    y = pickle.load(file)


                       
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Variabile per controllare se eseguire l'addestramento o meno
train_model = input('Eseguire addestramento? [Yes/No] ').lower()

while train_model not in ['y', 'n', 'yes', 'no']:
    print("Input non valido. inserire 'y', 'n', 'yes' o 'no'.")
    train_model = input("Eseguire addestramento? [Yes/No] ").lower()

if train_model == 'y' or train_model == 'yes':
    train_model = True
elif train_model == 'n' or train_model == 'no':
    train_model = False


if train_model:
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


################################ Ciclo di addestramento ###############################################

if train_model:


    # Train the model
    n_total_steps = len(train_dataloader)
    for epoch in range(max_epoch):
        for i, (images, labels) in enumerate(train_dataloader):  
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 28, 28]
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{max_epoch}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # Salva il modello addestrato
    model_save_path = 'mod_add_RNN.pth'
    torch.save(net.state_dict(),model_save_path)
else:
    model_save_path = 'mod_add_RNN.pth'

################################ Test Modello #############################################


# Carico modello
net=RNN(input_size=2400, 
          hidden_size=100, 
          num_layers=20, 
          output_size=6)
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Test set
inputs = torch.from_numpy(X_test).unsqueeze(1).float()
labels = torch.from_numpy(y_test).float()

test_dataset = TensorDataset(inputs, labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

dataiter = iter(test_dataloader)
#inputs, labels = next(dataiter)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    loss = 0
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss += criterion(outputs, labels).item()


    mse = loss / len(test_dataloader)
    print(f'Mean Square Error on the test set: {mse} %')

#
## Test 
#def test_accuracy(net, test_dataloader=test_dataloader):
#
#    with torch.no_grad():
#        predicted=[]
#        reals=[]
#        for data in test_dataloader:
#            inputs, real = data[0].to(device), data[1].to(device)
#            predict = net(inputs.to(device))
#            predicted.append(predict)
#            reals.append(real)
#
#    reals = torch.cat(reals, dim=0)
#    predicted = torch.cat(predicted, dim=0)
#
#    # get the accuracy for all value
#    errors = reals - predicted
#    errors= torch.Tensor.cpu(errors)
#    errors = torch.abs(errors)
#
#    # get best fitted curve
#    med_errors = torch.sum(errors, axis=1)
#    min_error = torch.min(med_errors)
#    index_min = torch.argmin(med_errors)
#    print("Errore minimo: ",min_error)
#    print(f'Assetto originale: {reals[index_min,:]}')
#    print(f'Assetto trovato: {predicted[index_min,:]}')
#
#    tollerance_velocity=0.0001
#    tollerance_position=1
#
#    # error like True or False
#    num_row, num_col = errors.size() 
#    errors_V = errors[:,0:3]
#    errors_P = errors[:,3:6]
#    boolean_eV = errors_V <= tollerance_velocity
#    boolean_eP = errors_P <= tollerance_position
#
#    float_tensor_V = boolean_eV.float()
#    float_tensor_P = boolean_eP.float()
#
#
#    accuracies_V = float_tensor_V.mean(dim=0)*100
#    accuracies_P = float_tensor_P.mean(dim=0)*100
#    accuracies_V=torch.Tensor.numpy(accuracies_V)
#    accuracies_P=torch.Tensor.numpy(accuracies_P)
#
#    return accuracies_V, accuracies_P
## Print accuracies
#
#accuracies_V, accuracies_P = test_accuracy(net,test_dataloader)
#print('testset:')
#for j in 0, 1, 2: 
#    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')
#
#print()
#for i in 0, 1, 2:
#    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')
#
#print()
#########
#accuracies_V, accuracies_P = test_accuracy(net,train_dataloader)
#print('trainset:')
#for j in 0, 1, 2: 
#    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')
#
#print()
#for i in 0, 1, 2:
#    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')
#
#print()
#


