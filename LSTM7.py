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
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from torch.nn.utils.rnn import pad_sequence

# Neural Network 
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        # Inizializza gli stati nascosti a zero all'inizio di ogni batch
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass attraverso l'LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Prendi solo l'output dell'ultimo passo temporale
        out = self.fc(out[:, -1, :])

        return out

# carico dataset 
def MyDataLoader(X, y, batch_size=64, window_size=200):
    windowed_data, windowed_labels = [], []
    for sequence, label in zip(X, y):
        # Divide la sequenza in finestre
        for i in range(0, len(sequence) - window_size + 1, window_size):
            window = sequence[i:i + window_size]
            windowed_data.append(window)
            windowed_labels.append(label)

    # Converti in tensori
    data_tensors = torch.tensor(windowed_data, dtype=torch.float32)
    label_tensors = torch.tensor(windowed_labels, dtype=torch.float32)

    # Crea il DataLoader
    dataset = TensorDataset(data_tensors, label_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

################################### main ###################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

with open("./dataCNN/X7", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y7", 'rb') as file:
    y = pickle.load(file)
     
y[:, :3] *= 10000
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
print('dimensione di X_train: ', X_train.shape)

train_dataloader = MyDataLoader(X_train, y_train)
test_dataloader = MyDataLoader(X_test, y_test)
val_dataloader = MyDataLoader(X_val, y_val)

# Definizione delle dimensioni degli strati
input_size = 1
hidden_size = 100  # Dimensione dell'hidden layer LSTM
output_size = 6  # Dimensione dell'output

net = LSTMNet(input_size, hidden_size, output_size).to(device)
print(net)

# Iperparametri
lr = 0.001
max_epoch = 200

# Definizione di loss function e optimizer
criterion = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)

# Inizializzazione di TensorBoard
writer = SummaryWriter('tensorboard/LSTM7')

# Ciclo di addestramento
for epoch in range(max_epoch):
    net.train()
    train_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward e optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calcola la norma dei gradienti
        total_norm = 0
        for p in net.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Registra la norma dei gradienti
        writer.add_scalar('Training/GradientNorm', total_norm, epoch)

    # Calcolo della loss media per l'epoca
    train_loss /= len(train_dataloader)
    writer.add_scalar('Training/Loss', train_loss, epoch)

    # Validazione
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_dataloader)
    writer.add_scalar('Validation/Loss', val_loss, epoch)

    print(f'Epoch [{epoch+1}/{max_epoch}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Chiudi il writer di TensorBoard dopo l'addestramento
writer.close()

# Salva il modello dopo l'addestramento
model_save_path='models/LSTM7.pth'
torch.save(net.state_dict(), model_save_path)


################################ Test Modello #############################################

# Assumi che model_save_path sia il percorso dove il modello è salvato
# Carico modello
net = LSTMNet(hidden_size=hidden_size, input_size=input_size, output_size=output_size)
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Test del modello
net.eval()
total_test_loss = 0
total_test_samples = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        total_test_loss += loss.item() * inputs.size(0)
        total_test_samples += inputs.size(0)

mse = total_test_loss / total_test_samples
print(f'Mean Square Error on the test set: {mse:.4f}')

# Funzione di test per la precisione
def test_accuracy(net, dataloader):
    net.eval()
    total_errors = []
    total_lengths = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)

            errors = torch.abs(labels - outputs)
            total_errors.append(errors)
            total_lengths += inputs.size(0)

    total_errors = torch.cat(total_errors, dim=0)
    avg_errors = total_errors.mean(dim=0)

    # Calcolo delle precisioni (modifica questi valori in base ai tuoi criteri specifici)
    tollerance_velocity = 0.0001 * 10000
    tollerance_position = 0.0174533
    accuracies_V = ((avg_errors[:3] <= tollerance_velocity).float().mean() * 100).item()
    accuracies_P = ((avg_errors[3:] <= tollerance_position).float().mean() * 100).item()

    return accuracies_V, accuracies_P

# Calcolo e stampa delle precisioni per il validation set
accuracies_V, accuracies_P = test_accuracy(net, train_dataloader)
print("Train set:")
print(f'Velocity accuracy: {accuracies_V:.2f} %')
print(f'Position accuracy: {accuracies_P:.2f} %')

# Calcolo e stampa delle precisioni per il validation set
accuracies_V, accuracies_P = test_accuracy(net, val_dataloader)
print("Validation set:")
print(f'Velocity accuracy: {accuracies_V:.2f} %')
print(f'Position accuracy: {accuracies_P:.2f} %')

# Calcolo e stampa delle precisioni per il test set
accuracies_V, accuracies_P = test_accuracy(net, test_dataloader)
print("Test set:")
print(f'Velocity accuracy: {accuracies_V:.2f} %')
print(f'Position accuracy: {accuracies_P:.2f} %')