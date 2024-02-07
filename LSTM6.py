import torch
import torch.nn as nn
import pickle
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################### carico dataset ##########################

# Funzione per applicare smoothing a una singola sequenza
def apply_smoothing(sequence, window_size=51, polyorder=3):
    # Assicurati che la sequenza sia abbastanza lunga per il filtro
    if len(sequence) >= window_size:
        return savgol_filter(sequence, window_size, polyorder)
    else:
        return sequence  # Restituisce la sequenza originale se troppo corta per lo smoothing

# Funzione per convertire le liste in tensori e applicare padding
def pad_and_convert_to_tensors(lists):
    # Applica prima lo smoothing a ciascuna sequenza
    smoothed_lists = [apply_smoothing(np.array(l)) for l in lists]
    tensors = [torch.tensor(l, dtype=torch.float32) for l in smoothed_lists]
    padded_tensors = pad_sequence(tensors, batch_first=True)
    return padded_tensors

class CustomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        length = len(sequence)
        label = self.labels[idx]
        return sequence, length, label


# Carica i dati
with open('./dataCNN/X41', 'rb') as f:
    X_lists = pickle.load(f)
with open('./dataCNN/y41', 'rb') as f:
    y = pickle.load(f)

# Applica smoothing e padding
X_padded = pad_and_convert_to_tensors(X_lists)

# Suddividi il dataset in train, validation e test set
X_train, X_temp, y_train, y_temp = train_test_split(X_padded, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

#lengths_train = [len(seq) for seq in X_train]
#lengths_train_tensor = torch.LongTensor(lengths_train)
#lengths_val = [len(seq) for seq in X_val]
#lengths_val_tensor = torch.LongTensor(lengths_val)

# Crea istanze di CustomDataset
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Crea i DataLoader
batch_size = 64  # Scegli una dimensione di batch appropriata
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

##################### Definire l'architettura della rete LSTM #####################
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Assicurati che x sia 3D (batch_size, seq_len, num_features)
        if x.dim() == 2:
            x = x.unsqueeze(2)

        # Inizializza gli stati nascosti a zero
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Impacchetta la sequenza
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(x_packed, (h0, c0))
        
        # Decomprimi il risultato
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :])  # Prende solo l'output dell'ultimo timestep
        return out

# Parametri del modello
input_size = 1  # Ogni timestep nel segnale è unidimensionale
hidden_size = 50  # Numero di unità nascoste
output_size = y_train.shape[1]  # Numero di output
num_layers = 2  # Numero di layer LSTM

model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# Definire la loss function e l'ottimizzatore
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Verifica se la GPU è disponibile e trasferisci il modello su GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Funzione per trasferire i tensori su GPU (se disponibile)
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Esempio di trasferimento dei dati di training su GPU
X_train, y_train = to_device(X_train, device), to_device(y_train, device)
X_val, y_val = to_device(X_val, device), to_device(y_val, device)

num_epochs = 1000
patience = 300 
loss_spann = []
loss_spann_val = []
best_loss = float('inf')
epochs_no_improve = 0

with open('losses.txt', 'w') as file:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, lenghts, y_batch in train_loader:  # Assumendo l'uso di un DataLoader
            
            # Aggiungi una dimensione per num_features se l'input è 1D
            if X_batch.dim() == 2:
                X_batch = X_batch.unsqueeze(2)  # Trasforma da (batch_size, seq_len) a (batch_size, seq_len, 1)

            X_batch, y_batch = to_device(X_batch, device), to_device(y_batch, device)
            
            # Impacchetta la sequenza
            X_packed = pack_padded_sequence(X_batch, lenghts, batch_first=True, enforce_sorted=False)
            
            optimizer.zero_grad()
            output = model(X_batch, lenghts)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            total_val_loss = 0
            total_samples = 0
            for X_batch, lenghts, y_batch in val_loader:  
                X_batch, y_batch = to_device(X_batch, device), to_device(y_batch, device)
                output = model(X_batch, lenghts)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                total_val_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)

            average_val_loss = total_val_loss / total_samples
            loss_spann_val.append(average_val_loss)

        val_loss /= len(val_loader.dataset)

        file.write(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n')

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

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
model_save_path = 'LSTM6.pth'
torch.save(model.state_dict(), model_save_path)

# Salva i log delle loss
with open('loss_spannLSTM6.txt', 'w') as file:
    for valore in loss_spann:
        file.write(str(valore) + '\n')

with open('loss_spannLSTM6_val.txt', 'w') as file:
    for valore in loss_spann_val:
        file.write(str(valore) + '\n')


################################ Test Modello #############################################

# Carico modello
net=LSTMModel(hidden_size=hidden_size, output_size=output_size)
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Train set

dataiter = iter(val_loader)
#inputs, labels = next(dataiter)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    loss = 0
    for input, labels, lengths in val_loader:
        input = input.to(device)
        labels = labels.to(device)
        outputs = net(input, lengths)
        loss += criterion(outputs, labels).item()


    mse = loss / len(train_loader)
    print(f'Mean Square Error on the valid set: {mse} %')


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

accuracies_V, accuracies_P = test_accuracy(net, train_loader)
print()
print('Train set:')
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')

print()
########
accuracies_V, accuracies_P = test_accuracy(net, val_loader)

print('Validation set:')
print()
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')

