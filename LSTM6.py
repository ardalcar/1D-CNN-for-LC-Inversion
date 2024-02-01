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
            for X_batch, lenghts, y_batch in val_loader:  # Assumendo l'uso di un DataLoader
                X_batch, y_batch = to_device(X_batch, device), to_device(y_batch, device)
                output = model(X_batch, lenghts)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        file.write(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n')

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
