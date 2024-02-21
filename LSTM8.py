import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Neural Network 
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, intermediate_size=128):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_intermediate = nn.Linear(hidden_size, intermediate_size)
        self.fc = nn.Linear(intermediate_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        
        # Inizializza gli stati nascosti a zero all'inizio di ogni batch
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0)) # Forward pass attraverso l'LSTM
        out = out[:, -1, :]  # Prendi solo l'output dell'ultimo passo temporale
        out = torch.tanh(self.fc_intermediate(out))
        out = torch.tanh(self.fc(out))

        return out

# Ciclo di addestramento
def learning(train_dataloader, val_dataloader, max_epoch):

    for epoch in range(max_epoch):
        net.train()
        train_loss = 0.0
        for j, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)

            if j == 10:
                
                labels_trov = net(inputs).detach()

                # Converti in array NumPy (sulla CPU e senza gradienti)
                labels10 = labels.cpu().numpy()
                labels_trov = labels_trov.cpu().numpy()

                # Denormalizza
                labels10 = denormalize_y(labels10)
                labels_trov = denormalize_y(labels_trov)

                # Assicurati che labels e labels_trov abbiano la stessa dimensione e siano 1D
                for i in range(labels10.shape[0]):
                    for k in range(labels10.shape[1]):
                        writer.add_scalars(f'Training/Labels10/Sample_{i}_Feature_{k}', 
                                          {'Real': labels10[i, k],
                                           'Network': labels_trov[i, k]}, epoch)

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
            j+=1

       
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
    model_save_path='models/LSTM8.pth'
    torch.save(net.state_dict(), model_save_path)

def tensor_to_array(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Se il tensore richiede gradienti, prima crea una copia del tensore senza gradienti
    if tensor.requires_grad:
        # Usa detach() per ottenere una copia senza informazioni di gradiente
        tensor = tensor.detach()

    # Converti il tensore PyTorch in un array NumPy
    array = tensor.numpy()
    return array

def truncate_to_shortest_and_convert_to_array(light_curves):
    # Trova la lunghezza della curva più corta
    min_length = min(len(curve) for curve in light_curves)

    # Tronca tutte le curve alla lunghezza della curva più corta e le converte in un array
    truncated_curves = [curve[:min_length] for curve in light_curves]
    array_curves = np.array(truncated_curves)

    return array_curves

def normalize_array(Input, max, min):
    norm_arr = (Input - min) / (max - min)
    return norm_arr

def normalize_y(y, max_angle=1.5, min_angle=-1.5, max_vel=0.0002, min_vel=-0.0002):
    y_angle=y[:,-3:]
    y_vel=y[:,:3]
    y_norm_angle=normalize_array(y_angle, max_angle, min_angle)
    y_norm_vel=normalize_array(y_vel, max_vel, min_vel)
    y_norm=np.concatenate((y_norm_vel, y_norm_angle), axis=1)
    return y_norm

def denormalize_array(norm_arr, max, min):
    input_arr = norm_arr * (max - min) + min
    return input_arr

def denormalize_y(y_norm, max_angle=1.5, min_angle=-1.5, max_vel=0.0002, min_vel=-0.0002):
    y_norm_vel = y_norm[:, :3]
    y_norm_angle = y_norm[:, -3:]
    
    y_vel = denormalize_array(y_norm_vel, max_vel, min_vel)
    y_angle = denormalize_array(y_norm_angle, max_angle, min_angle)

    y = np.concatenate((y_vel, y_angle), axis=1)
    return y

# carico dataset 
def MyDataLoader(X, y, batch_size=64, shuffle=True):
    # Converti in tensori
    data_tensors = torch.tensor(X, dtype=torch.float32)
    label_tensors = torch.tensor(y, dtype=torch.float32)

    # Crea il DataLoader
    dataset = TensorDataset(data_tensors, label_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return  dataset, dataloader

# Funzione di test per la precisione
def test_accuracy(net, dataloader):
    net.eval()
    total_errors = []
    total_lengths = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            labels = denormalize_array(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            errors = torch.abs(labels - outputs)
            total_errors.append(errors)
            total_lengths += inputs.size(0)

    total_errors = torch.cat(total_errors, dim=0)
    avg_errors = total_errors.mean(dim=0)

    # Calcolo delle precisioni 
    tollerance_velocity = 0.0001
    tollerance_position = 0.0174533
    accuracies_V = ((avg_errors[:3] <= tollerance_velocity).float().mean() * 100).item()
    accuracies_P = ((avg_errors[3:] <= tollerance_position).float().mean() * 100).item()

    return accuracies_V, accuracies_P

################################### main ###################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

with open("./dataCNN/X7", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y7", 'rb') as file:
    y = pickle.load(file)

X = truncate_to_shortest_and_convert_to_array(X)

y = normalize_y(y)
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

train_dataset, train_dataloader = MyDataLoader(X_train, y_train, batch_size=1, shuffle=False)
val_dataset, val_dataloader = MyDataLoader(X_val, y_val, batch_size=1, shuffle=False)
test_dataset, test_dataloader = MyDataLoader(X_test, y_test, batch_size=1, shuffle=False)

# Definizione delle dimensioni degli strati
input_size = 1
hidden_size = 100  # Dimensione dell'hidden layer LSTM
output_size = 6  # Dimensione dell'output
num_layers=2 
intermediate_size=64

net = LSTMNet(input_size, hidden_size, output_size, num_layers, intermediate_size).to(device)
print(net)

# Iperparametri
lr = 0.001
max_epoch = 1000

# Definizione di loss function e optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0001)

# Inizializzazione di TensorBoard
writer = SummaryWriter('tensorboard/LSTM8')

learning(train_dataloader, val_dataloader, max_epoch)

################################ Test Modello #############################################

model_save_path='models/LSTM8.pth'

# Carico modello
net = LSTMNet(input_size, hidden_size, output_size, num_layers, intermediate_size)
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