import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split

# Definizione dell'Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, encoding_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, encoding_size),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Funzione per applicare il windowing
def apply_windowing(data, window_size):
    windowed_data = []
    for sequence in data:
        if len(sequence) >= window_size:
            for i in range(0, len(sequence) - window_size + 1, window_size):
                windowed_data.append(sequence[i:i + window_size])
    return windowed_data

def MyDataLoader(X):
    # Applica il windowing
    window_size = 200
    windowed_curves = apply_windowing(X, window_size)
    # Applica il padding
    padded_windows = pad_sequence([torch.tensor(window) for window in windowed_curves], batch_first=True, padding_value=0).numpy()
    # Riduzione della dimensionalità con PCA
    pca = PCA(n_components=50)  # Adatta questo valore
    X_reduced = pca.fit_transform(padded_windows.reshape(len(padded_windows), -1))
    # Rimozione degli outlier con DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    clusters = dbscan.fit_predict(X_reduced)
    non_outliers = X_reduced[clusters != -1]
    num_outliers = np.sum(clusters == -1)
    num_non_outliers = np.sum(clusters != -1)
    print("Numero di punti non outlier:", num_non_outliers)
    print("Numero di outlier identificati e rimossi:", num_outliers)
    # Preparazione del DataLoader
    data_tensors = torch.tensor(non_outliers, dtype=torch.float32)
    dataset = TensorDataset(data_tensors)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, pca

def train(model, optimizer, dataloader, criterion, device, num_epochs=10):
    model.train()
    total_loss = 0

    for epoch in range(num_epochs):
        for inputs in dataloader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

def validate(model, dataloader, criterion, device):
    model.eval()  # Imposta il modello in modalità di valutazione
    total_loss = 0

    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for inputs in dataloader:
            inputs = inputs[0].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            total_loss += loss.item() * inputs.size(0)

    average_loss = total_loss / len(dataloader.dataset)
    return average_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Carica i dati
with open("./dataCNN/X41", 'rb') as file:
    X = pickle.load(file)

X_train, X_val = train_test_split(X, test_size=0.3, random_state=42)
train_dataloader, pca_train = MyDataLoader(X_train)
val_dataloader, pca_val = MyDataLoader(X_val)

input_size = 50

# Definisci lo spazio degli iperparametri
space = [
    Integer(32, 256, name='hidden_size'),
    Integer(16, 128, name='encoding_size'),
    Real(1e-4, 1e-2, "log-uniform", name='learning_rate')
]

criterion=nn.MSELoss()
# Funzione obiettivo per l'ottimizzazione bayesiana
@use_named_args(space)
def objective(**params):
    autoencoder = Autoencoder(input_size, params['hidden_size'], params['encoding_size']).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=params['learning_rate'])
    
    # Addestra l'autoencoder con questi parametri
    train_loss = train(autoencoder, optimizer, train_dataloader, criterion, device)  # Assumi che questa funzione sia definita
    val_loss = validate(autoencoder, val_dataloader, criterion, device)  # Assumi che questa funzione sia definita
    
    # Restituisci l'errore di valutazione (da minimizzare)
    return val_loss

# Esegui l'ottimizzazione bayesiana
results = gp_minimize(objective, space, n_calls=15, random_state=0)

print("Migliori parametri:", results.x)
