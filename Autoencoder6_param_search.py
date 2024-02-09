import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split
from Autoencoder6 import Autoencoder,  MyDataLoader # Assumi che queste funzioni siano definite nel tuo script Autoencoder6.py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open("./dataCNN/X41", 'rb') as file:
    X_temp = pickle.load(file)

X_train, X_val = train_test_split(X_temp, test_size=0.3, random_state=42)
train_dataloader = MyDataLoader(X_train)
val_dataloader = MyDataLoader(X_val)
input_size = 50

def train(model, optimizer, dataloader, criterion, device):
    model.train()
    total_loss = 0

    for inputs in dataloader:
        inputs = inputs[0].to(device)  # Assicurati che 'inputs' sia corretto

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    average_loss = total_loss / len(dataloader.dataset)
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
    autoencoder = Autoencoder(input_size, params['hidden_size'], params['encoding_size'])
    optimizer = optim.Adam(autoencoder.parameters(), lr=params['learning_rate'])
    
    # Addestra l'autoencoder con questi parametri
    train_loss = train(autoencoder, optimizer, train_dataloader, criterion, device)  # Assumi che questa funzione sia definita
    val_loss = validate(autoencoder, val_dataloader, criterion, device)  # Assumi che questa funzione sia definita
    
    # Restituisci l'errore di valutazione (da minimizzare)
    return val_loss

# Esegui l'ottimizzazione bayesiana
results = gp_minimize(objective, space, n_calls=15, random_state=0)

print("Migliori parametri:", results.x)
