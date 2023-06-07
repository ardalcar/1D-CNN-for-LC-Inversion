import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Imposta il seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Carica il database da file
X = np.load('X.npy')
y = np.load('y.npy')

# Dividi il dataset in addestramento e verifica in modo casuale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

# Converti i dati di input in tensori di PyTorch
inputs = torch.from_numpy(X_train).unsqueeze(1).float()
labels = torch.from_numpy(y_train).float()

# Crea il modello della rete neurale
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2400 * 32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 6)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Crea un'istanza del modello
model = NeuralNetwork()

# Definisci l'ottimizzatore e la funzione di loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Addestramento del modello
num_epochs = 10
batch_size = 32
num_batches = len(inputs) // batch_size

for epoch in range(num_epochs):
    total_loss = 0.0
    
    # Divisione dei dati in mini-batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch_inputs = inputs[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calcolo della loss media per epoch
    average_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")

# Valutazione del modello sui dati di test
test_inputs = torch.from_numpy(X_test).unsqueeze(1).float()
test_labels = torch.from_numpy(y_test).float()

with torch.no_grad():
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_labels)

print(f"Test Loss: {test_loss.item():.4f}")
