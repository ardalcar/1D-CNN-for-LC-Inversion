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
from RNN3 import test_accuracy as ta


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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




# Creazione dell'istanza della rete neurale
hidden_sizes = [64, 128, 256, 512, 1024]  # Ad esempio, valori da testare
output_size = 6  # Dimensione dell'output
max_epoch = 100
batch_sizes = [16, 32, 64, 128, 256, 512] 
criterion = nn.MSELoss().device()

##################################### carico dataset ##########################

with open("./dataCNN/X3", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y3", 'rb') as file:
    y = pickle.load(file)

# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_train], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_train).float()

lengths_train = [len(seq) for seq in X_train]
lengths_train_tensor = torch.LongTensor(lengths_train)
i=0

for hidden_size in hidden_sizes:
    for batch_size in batch_sizes:
        i+=1
        train_dataset = TensorDataset(inputs, labels, lengths_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Test set
        lengths_test = [len(seq) for seq in X_test]
        lengths_test_tensor = torch.LongTensor(lengths_test)

        inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_test], batch_first=True, padding_value=0)
        labels = torch.from_numpy(y_test).float()

        test_dataset = TensorDataset(inputs, labels, lengths_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        results = []

        net = RNN(hidden_size, output_size)
        net = nn.DataParallel(RNN)
        net.to(device)
        optimizer = optim.Adam(net.parameters()) 

        for epoch in range(max_epoch):
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                # Backward e optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Valuta il modello dopo l'allenamento
        # Potresti voler eseguire il modello sui dati di validazione/test e calcolare la metrica desiderata
        results.append((hidden_size, loss.item()))
        print(f'test {i} done')

# Stampa o salva i risultati
for hidden_size, performance in results:
    print(f"Hidden Size: {hidden_size}, Performance: {loss.item()}")

# Potresti voler salvare i risultati in un file o in un formato più strutturato
