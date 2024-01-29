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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

################################## Neural Network ################################

class LSTMNet(nn.Module):
    def __init__(self, hidden_size, output_size=6):  # Assicurati che output_size sia 6 se il tuo target ha 6 caratteristiche
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, x, lengths):

        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)

        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h,c) = self.lstm(packed_input)
        out = h[-1]  # Ultimo layer dell'output dello stato nascosto
        out = out.view(out.size(0), -1)  # Modifica la forma se necessario
        out = self.fc(out)
        
        return out

# Definizione delle dimensioni degli strati
hidden_size = 100  # Dimensione dell'hidden layer LSTM
output_size = 6  # Dimensione dell'output

# Creazione dell'istanza della rete neurale
net = LSTMNet(hidden_size, output_size)
net.to(device)

# Stampa dell'architettura della rete
print(net)

# iperparametri
lr = 0.001        # learning rate
momentum = 0.001  # momentum
max_epoch = 2000  # numero di epoche
batch_size = 128  # batch size
scaler = GradScaler()

criterion = nn.L1Loss().to(device)
#optimizer = optim.SGD(net.parameters(), lr)
optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=0.0001) # Regularizzazione L2 (Weight Decay)



##################################### carico dataset ##########################

with open("./dataCNN/X41", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y41", 'rb') as file:
    y = pickle.load(file)

with open("./dataCNN/X41", 'rb') as file:
    X1 = pickle.load(file)

with open("./dataCNN/y41", 'rb') as file:
    y1 = pickle.load(file)
     
                       
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X_train, X_temp, y_train, y_temp = train_test_split(X1, y1, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
X_train=X
y_train=y

## Train set
inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_train], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_train).float()

lengths_train = [len(seq) for seq in X_train]
lengths_train_tensor = torch.LongTensor(lengths_train)

train_dataset = TensorDataset(inputs, labels, lengths_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

## Validation test
inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_val], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_val).float()

lengths_val = [len(seq) for seq in X_val]
lengths_val_tensor = torch.LongTensor(lengths_val)

val_dataset = TensorDataset(inputs, labels, lengths_val_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


## Test set
lengths_test = [len(seq) for seq in X_test]
lengths_test_tensor = torch.LongTensor(lengths_test)

inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_test], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_test).float()

test_dataset = TensorDataset(inputs, labels, lengths_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

################################ Ciclo di addestramento ###############################################

writer = SummaryWriter('tensorboard/LSTM')
loss_spann = []
loss_spann_val = []  # Per tenere traccia della loss sul validation set

patience = 300  # Numero di epoche da attendere dopo l'ultimo miglioramento
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(max_epoch):
    # Training loop
    for i, (input, labels, lengths) in enumerate(train_dataloader):  
        input = input.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = net(input, lengths)
        outputs = outputs.squeeze(0) 
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()

    # Calcolo della loss sul validation set
    with torch.no_grad():
        total_val_loss = 0
        total_samples = 0
        for input_val, labels_val, lengths_val in val_dataloader:
            input_val = input_val.to(device)
            labels_val = labels_val.to(device)

            outputs_val = net(input_val, lengths_val)
            loss_val = criterion(outputs_val, labels_val)

            total_val_loss += loss_val.item() * len(labels_val)
            total_samples += len(labels_val)

        average_val_loss = total_val_loss / total_samples
        loss_spann_val.append(average_val_loss)

    writer.add_scalar('Loss/Train', loss.item(), epoch)
    writer.add_scalar('Loss/Validation', average_val_loss, epoch)

    print(f'Epoch [{epoch+1}/{max_epoch}] Loss: {loss.item():.4f} Loss validation: {average_val_loss:.4f}')
    loss_spann.append(loss.item())

    # Controllo per l'early stopping
    if average_val_loss < best_loss:
        best_loss = average_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print("Early stopping triggered")
        break


# Salva il modello addestrato
model_save_path = 'LSTM5.pth'
torch.save(net.state_dict(), model_save_path)

# Salva i log delle loss
with open('loss_spannLSTM5.txt', 'w') as file:
    for valore in loss_spann:
        file.write(str(valore) + '\n')

with open('loss_spannLSTM5_val.txt', 'w') as file:
    for valore in loss_spann_val:
        file.write(str(valore) + '\n')


################################ Test Modello #############################################

# Carico modello
net=LSTMNet(hidden_size=hidden_size, output_size=output_size)
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Test set

dataiter = iter(test_dataloader)
#inputs, labels = next(dataiter)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    loss = 0
    for input, labels, lengths in test_dataloader:
        input = input.to(device)
        labels = labels.to(device)
        outputs = net(input, lengths)
        loss += criterion(outputs, labels).item()


    mse = loss / len(test_dataloader)
    print(f'Mean Square Error on the test set: {mse} %')


# Test 
def test_accuracy(net, test_dataloader):
    net.eval()  # Imposta la rete in modalità valutazione
    predicted=[]
    reals=[]
    with torch.no_grad():
        for data in test_dataloader:
            inputs, real, lengths = data
            inputs, real = inputs.to(device), real.to(device)
            output = net(inputs, lengths)
            predicted.append(output)
            reals.append(real)

        # Concatena i tensori, gestendo separatamente l'ultimo batch se necessario
        predicted = torch.cat([p for p in predicted], dim=0)
        reals = torch.cat([r for r in reals], dim=0)

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

print()
########
accuracies_V, accuracies_P = test_accuracy(net,val_dataloader)
print('trainset:')
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')
