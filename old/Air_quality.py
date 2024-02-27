import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Verifica se CUDA (GPU) è disponibile, altrimenti usa la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importa il dataset
heart_disease = fetch_ucirepo(id=45)
X, y = heart_disease.data.features, heart_disease.data.targets

# Converti X in DataFrame di pandas se necessario
X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

# Converti y in Series di pandas se necessario
if isinstance(y, pd.DataFrame):
    # Se y è un DataFrame, assicurati che sia una singola colonna e convertila in Series
    y = y.iloc[:, 0] if y.shape[1] == 1 else y
elif not isinstance(y, pd.Series):
    # Se y non è né un DataFrame né una Series, convertila in Series
    y = pd.Series(y)

# Normalizza i dati
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Dividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Converti in tensori PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

# Definizione del modello di rete neurale
class RegressionTransformer(nn.Module):
    def __init__(self, input_size, num_heads, dim_feedforward, num_layers):
        super(RegressionTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, dim_feedforward)
        encoder_layers = TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(dim_feedforward, 1)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        output = self.output_layer(output)
        return output.squeeze()

# Crea il modello e spostalo sul dispositivo appropriato
model = RegressionTransformer(input_size=X_train.shape[1], num_heads=4, dim_feedforward=128, num_layers=2)
model.to(device)

# Definisci la funzione di perdita e l'ottimizzatore
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Addestramento del modello
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Sposta i tensori sul dispositivo appropriato
    X_train_tensor = X_train.to(device)
    y_train_tensor = y_train.to(device)

    # Forward pass
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    # Verifica la presenza di nan nella loss
    if torch.isnan(loss):
        print(f"Loss is nan at epoch {epoch+1}")
        break

    # Backward pass e aggiornamento dei pesi
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Valutazione del modello
model.eval()
with torch.no_grad():
    X_test_tensor = X_test.to(device)
    y_test_tensor = y_test.to(device)
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
