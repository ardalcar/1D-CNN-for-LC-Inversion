import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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

# Carica i dati
with open("./dataCNN/X41", 'rb') as file:
    X = pickle.load(file)

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

dataloader, pca = MyDataLoader(X)

# Configurazione dell'addestramento dell'Autoencoder
autoencoder = Autoencoder(50, 25, 12).to(device)  # Adatta queste dimensioni
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Ciclo di addestramento
num_epochs = 1
loss_spann=[]
for epoch in range(num_epochs):
    for inputs in dataloader:
        inputs = inputs[0].to(device)
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    loss_spann.append(loss.item())

# Salva il modello addestrato
torch.save(autoencoder.state_dict(), 'autoencoder6.pth')

with open('loss_spannAutoencoder6.txt', 'w') as file:
    for valore in loss_spann:
        file.write(str(valore) + '\n')

############## preprocess ################

def preprocess_single_sample(sample, pca, window_size=200, input_size=50):
    """
    Applica windowing, padding e PCA a un singolo campione.

    :param sample: Il campione da preprocessare.
    :param pca: L'oggetto PCA già addestrato sul dataset completo.
    :param window_size: La dimensione della finestra per il windowing.
    :param input_size: La dimensione dell'input dopo il padding.
    :return: Campione preprocessato.
    """
    # Applica il windowing
    windowed_sample = []
    for i in range(0, len(sample) - window_size + 1, window_size):
        windowed_sample.append(sample[i:i + window_size])
    
    # Applica il padding
    padded_sample = [np.pad(window, (0, window_size - len(window)), 'constant', constant_values=0) for window in windowed_sample]
    
    # Applica la PCA
    sample_reduced = pca.transform(np.array(padded_sample).reshape(len(padded_sample), -1))

    return sample_reduced

# Utilizzo della funzione
# Assicurati di passare l'oggetto PCA che hai già addestrato
sample_index = 0  # Indice del campione da valutare
sample = X[sample_index]
processed_sample = preprocess_single_sample(sample, pca)

processed_sample_tensor = torch.tensor(processed_sample).float().to(device).unsqueeze(0)
with torch.no_grad():
    reconstructed_sample = autoencoder(processed_sample_tensor)

# Confronta il campione originale con quello ricostruito
reconstructed_sample_np = reconstructed_sample.cpu().numpy().squeeze(0)

# Visualizzazione (adatta a seconda della forma e del tipo dei tuoi dati)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Input Originale")
plt.plot(sample)  # o plt.imshow(...) per dati immagine

plt.subplot(1, 2, 2)
plt.title("Output Ricostruito")
plt.plot(reconstructed_sample_np)  # o plt.imshow(...) per dati immagine
plt.savefig('Autoencoder.png')
#plt.show()