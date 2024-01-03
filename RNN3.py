import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence

#device = (
#    f"cuda:0"
#    if torch.cuda.is_available()
#    else "mps"
#    if torch.backends.mps.is_available()
#    else "cpu"
#)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

################################## Neural Network ################################

class RNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RNN, self).__init__()
       
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)       
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # LSTM forward pass
        _, (hidden, _) = self.lstm(packed_input)
        # Use the last hidden state
        out = self.fc(hidden[-1])
        return out

# Definizione delle dimensioni degli strati
hidden_size = 128  # Dimensione dell'hidden layer LSTM
output_size = 6  # Dimensione dell'output

# Creazione dell'istanza della rete neurale
net = RNN(hidden_size, output_size)
#net = nn.DataParallel(RNN)
net.to(device)

# Stampa dell'architettura della rete
print(net)


# iperparametri
lr = 0.2          # learning rate
momentum = 0.001  # momentum
max_epoch = 3000   # numero di epoche
batch_size = 20   # batch size
scaler = GradScaler()


criterion = nn.MSELoss().to(device)
#optimizer = optim.Adam(net.parameters(), lr)
optimizer = optim.Adam(net.parameters(), lr)


##################################### carico dataset ##########################

with open("./dataCNN/X3", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y3", 'rb') as file:
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
    inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_train], batch_first=True, padding_value=0)
    #inputs = torch.from_numpy(X_train).unsqueeze(1).float()
    labels = torch.from_numpy(y_train).float()
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_train], batch_first=True, padding_value=0)
    labels = torch.from_numpy(y_train).float()

lengths_train = [len(seq) for seq in X_train]
lengths_train_tensor = torch.LongTensor(lengths_train)

train_dataset = TensorDataset(inputs, labels, lengths_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Test set
lengths_test = [len(seq) for seq in X_test]
lengths_test_tensor = torch.LongTensor(lengths_test)

inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_test], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_test).float()

test_dataset = TensorDataset(inputs, labels, lengths_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



################################ Ciclo di addestramento ###############################################

if train_model:

    loss_spann=[]
    loss_spann_test=[]
    # Train the model
    n_total_steps = len(train_dataloader)
    max_norm=5 #gradient clipping
    for epoch in range(max_epoch):
        for i, (images, labels, lengths) in enumerate(train_dataloader):  
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 28, 28]
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images, lengths)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            total_norm = 0
            for param in net.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"Epoch: {epoch}, Gradient Norm: {total_norm}")

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            optimizer.step()
    
        # Calcolo della loss sul test set
        with torch.no_grad():
            total_test_loss = 0
            total_samples = 0
            for images_test, labels_test, lengths_test in test_dataloader:
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = net(images_test, lengths_test)
                loss_test = criterion(outputs_test, labels_test)

                total_test_loss += loss_test.item() * len(labels_test)
                total_samples += len(labels_test)

            average_test_loss = total_test_loss / total_samples
            loss_spann_test.append(average_test_loss)

        print (f'Epoch [{epoch+1}/{max_epoch}] Loss: {loss.item():.4f} Loss test: {loss_test.item():.4f}')
        loss_spann.append(loss.item())

    # Salva il modello addestrato
    model_save_path = 'RNN3.pth'
    torch.save(net.state_dict(),model_save_path)

    with open('loss_spannRNN3.txt','w') as file:
        for valore in loss_spann:
            file.write(str(valore) + '\n')

    with open('loss_spannRNN3_test.txt', 'w') as file:
        for valore in loss_spann_test:
            file.write(str(valore) + '\n')


else:
    model_save_path = 'RNN3.pth'


################################ Test Modello #############################################


# Carico modello
net=RNN(hidden_size=hidden_size, 
        output_size=output_size)
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Test set

dataiter = iter(test_dataloader)
#inputs, labels = next(dataiter)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    loss = 0
    for images, labels, lengths in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images, lengths)
        loss += criterion(outputs, labels).item()


    mse = loss / len(test_dataloader)
    print(f'Mean Square Error on the test set: {mse} %')


# Test 
def test_accuracy(net, test_dataloader=test_dataloader):

    with torch.no_grad():
        predicted=[]
        reals=[]
        for data in test_dataloader:
            inputs, real = data[0].to(device), data[1].to(device)
            predict = net(inputs.to(device))
            predicted.append(predict)
            reals.append(real)

    reals = torch.cat(reals, dim=0)
    predicted = torch.cat(predicted, dim=0)

    # get the accuracy for all value
    errors = reals - predicted
    errors= torch.Tensor.cpu(errors)
    errors = torch.abs(errors)

    # get best fitted curve
    med_errors = torch.sum(errors, axis=1)
    min_error = torch.min(med_errors)
    index_min = torch.argmin(med_errors)
    print("Errore minimo: ",min_error)
    print(f'Assetto originale: {reals[index_min,:]}')
    print(f'Assetto trovato: {predicted[index_min,:]}')

    tollerance_velocity=0.0001
    tollerance_position=0.0174533

    # error like True or False
    errors_V = errors[:,0:3]
    errors_P = errors[:,3:6]
    boolean_eV = errors_V <= tollerance_velocity
    boolean_eP = errors_P <= tollerance_position

    float_tensor_V = boolean_eV.float()
    float_tensor_P = boolean_eP.float()


    accuracies_V = float_tensor_V.mean(dim=0)*100
    accuracies_P = float_tensor_P.mean(dim=0)*100
    accuracies_V=torch.Tensor.numpy(accuracies_V)
    accuracies_P=torch.Tensor.numpy(accuracies_P)

    return accuracies_V, accuracies_P
# Print accuracies

accuracies_V, accuracies_P = test_accuracy(net,test_dataloader)
print('testset:')
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')

print()
########
accuracies_V, accuracies_P = test_accuracy(net,train_dataloader)
print('trainset:')
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')
