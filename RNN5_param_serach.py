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
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

################################## Neural Network ################################

class RNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RNN, self).__init__()
       
        self.hidden_size = hidden_size

        # Definisci il layer LSTM
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        
        # Definisci il layer Fully Connected
        self.fc = nn.Linear(hidden_size, output_size)

        # Aggiungi uno strato di Batch Normalization
        # La dimensione '1' corrisponde a 'input_size' della LSTM
        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, x, lengths):
        # Applica Batch Normalization
        # x ha dimensioni (batch, seq_len, features), BatchNorm1d si aspetta (batch, features, seq_len)
        x = x.transpose(1, 2)  # Scambia seq_len e feature
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # Riporta alla forma originale

        # Pack padded sequence
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass 
        _, (hidden, _) = self.lstm(packed_input)
        
        # Use the last hidden state
        out = self.fc(hidden[-1])
        
        return out

# Definizione delle dimensioni degli strati
hidden_size = 32  # Dimensione dell'hidden layer LSTM
output_size = 6  # Dimensione dell'output

# Creazione dell'istanza della rete neurale
net = RNN(hidden_size, output_size)
net.to(device)

# Stampa dell'architettura della rete
print(net)

# iperparametri
lr = 0.0001          # learning rate
momentum = 0.001  # momentum
max_epoch = 1000    # numero di epoche
batch_size = 128  # batch size
scaler = GradScaler()

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr)


##################################### carico dataset ##########################

with open("./dataCNN/X3", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y3", 'rb') as file:
    y = pickle.load(file)
    
                       
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

r=0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=seed)
z2=len(X_train)
z3=len(X_test)
print(f'Il trainset contiene {z2} samples')
print(f'Il testset contiene {z3} samples')
inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_train], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_train).float()

lengths_train = [len(seq) for seq in X_train]
lengths_train_tensor = torch.LongTensor(lengths_train)

train_dataset = TensorDataset(inputs, labels, lengths_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Test set
lengths_test = [len(seq) for seq in X_test]
lengths_test_tensor = torch.LongTensor(lengths_test)

inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_test], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_test).float()

test_dataset = TensorDataset(inputs, labels, lengths_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

######################## Ricerca booleana ########################

# Spazi degli iperparametri
space  = [
    Real(1e-6, 1e-1, "log-uniform", name='learning_rate'),
    Integer(10, 100, name='hidden_size'),
    Categorical([nn.MSELoss, nn.L1Loss], name='loss_function'),
    Categorical([torch.optim.Adam, torch.optim.SGD], name='optimizer')
]

# Funzione obiettivo per l'ottimizzazione
@use_named_args(space)
def objective(learning_rate, hidden_size, loss_function, optimizer):
    # Converti hidden_size in un intero Python standard
    hidden_size = int(hidden_size)
    model = RNN(hidden_size=hidden_size, output_size=output_size)
    model.to(device)

    if optimizer == torch.optim.Adam:
        optimizer = optimizer(model.parameters(), lr=learning_rate)
    elif optimizer == torch.optim.SGD:
        optimizer = optimizer(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = loss_function()

    loss_spann=[]
    loss_spann_test=[]
    # Train the model
    n_total_steps = len(train_dataloader)
    max_norm=50 #gradient clipping

    for epoch in range(max_epoch):
        for i, (images, labels, lengths) in enumerate(train_dataloader):  

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images, lengths) 
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
    
        # Calcolo della loss sul test set
        with torch.no_grad():
            total_test_loss = 0
            total_samples = 0
            for images_test, labels_test, lengths_test in test_dataloader:
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = net(images_test, lengths_test)
                loss_test = criterion(outputs_test, labels_test)

                total_test_loss += loss_test.item() * len(labels_test)
                total_samples += len(labels_test)

            average_test_loss = total_test_loss / total_samples
            loss_spann_test.append(average_test_loss)

        print (f'Epoch [{epoch+1}/{max_epoch}] Loss: {loss.item():.4f} Loss test: {loss_test.item():.4f}')
        loss_spann.append(loss.item())
        validation_loss=0.7*loss.item()+0.3*loss_test.item()

    return validation_loss

res_gp = gp_minimize(objective, space, n_calls=20, random_state=0)
print("Migliori parametri: %s" % res_gp.x)
