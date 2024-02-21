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

# Learning cicle
def learning(train_dataloader, val_dataloader, max_epoch):

    for epoch in range(max_epoch):
        net.train()
        train_loss = 0.0
        j=0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
           
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            if j == 1:
                specific_output = outputs[9]  # Decimo elemento del decimo batch
                specific_label = labels[9]

                # Converti in array NumPy (sulla CPU e senza gradienti)
                specific_output = specific_output.detach().cpu().numpy()
                specific_label = specific_label.detach().cpu().numpy()

                # Assicurati che siano bidimensionali prima di denormalizzare
                specific_output_reshaped = specific_output[np.newaxis, :]
                specific_label_reshaped = specific_label[np.newaxis, :]

                specific_output_denorm = denormalize_y(specific_output_reshaped)
                specific_label_denorm = denormalize_y(specific_label_reshaped)
                print(f'label original : {specific_label_denorm[0][0]*1e4:.3f} 1e-4 '
                      f'{specific_label_denorm[0][1]*1e4:.3f} 1e-4 '
                      f'{specific_label_denorm[0][2]*1e4:.3f} 1e-4 '
                      f'{specific_label_denorm[0][3]:.3f} '
                      f'{specific_label_denorm[0][4]:.3f} '
                      f'{specific_label_denorm[0][5]:.3f}')
                print(f'output nn      : {specific_output_denorm[0][0]*1e4:.3f} 1e-4 '
                      f'{specific_output_denorm[0][1]*1e4:.3f} 1e-4 '
                      f'{specific_output_denorm[0][2]*1e4:.3f} 1e-4 '
                      f'{specific_output_denorm[0][3]:.3f} '
                      f'{specific_output_denorm[0][4]:.3f} '
                      f'{specific_output_denorm[0][5]:.3f}')

                # Registra su TensorBoard
                for j in range(specific_output_denorm.shape[1]):
                    writer.add_scalar(f'Training/Predicted_Feature_{j}', specific_output_denorm[0, j], epoch)
                    writer.add_scalar(f'Training/Actual_Feature_{j}', specific_label_denorm[0, j], epoch)
            
            j+=1
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)



        # Calcolo della loss media per la validazione
        # Validazione
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                val_outputs = net(inputs)
                loss = criterion(val_outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        print(f'Epoch [{epoch+1}/{max_epoch}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print()
        writer.add_scalar('Training/TrainLoss', train_loss, epoch)
        writer.add_scalar('Training/ValLoss', val_loss, epoch)

    writer.close()
    model_save_path = 'models/LSTM8.pth'
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
    if len(y_norm.shape) == 1:
        y_norm = y_norm[np.newaxis, :]
        a=True
    else:
        a=False

    y_norm_vel = y_norm[:, :3]
    y_norm_angle = y_norm[:, -3:]
    
    y_vel = denormalize_array(y_norm_vel, max_vel, min_vel)
    y_angle = denormalize_array(y_norm_angle, max_angle, min_angle)

    y = np.concatenate((y_vel, y_angle), axis=1)
    if a:
        y = y.squeeze(0)
    return y

# carico dataset 
def MyDataLoader(X, y,  batch_size):
    # Converti in tensori
    data_tensors = torch.tensor(X, dtype=torch.float32)
    label_tensors = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(data_tensors, label_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return  dataloader

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

batch_size = 100
train_dataloader = MyDataLoader(X_train, y_train, batch_size)
val_dataloader = MyDataLoader(X_val, y_val, batch_size)
test_dataloader = MyDataLoader(X_test, y_test, batch_size)

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
batch_size = 200

# Definizione di loss function e optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0001)

# Inizializzazione di TensorBoard
writer = SummaryWriter('tensorboard/LSTM8')

learning(train_dataloader, val_dataloader, max_epoch)

################################ Test Modello #############################################

#model_save_path='models/LSTM8.pth'
#
## Carico modello
#net = LSTMNet(input_size, hidden_size, output_size, num_layers, intermediate_size)
#net.to(device)
#net.load_state_dict(torch.load(model_save_path))
#
## Test del modello
#net.eval()
#total_test_loss = 0
#total_test_samples = 0
#
#with torch.no_grad():
#    for inputs, labels in test_dataloader:
#        inputs, labels = inputs.to(device), labels.to(device)
#
#        outputs = net(inputs)
#
#        loss = criterion(outputs, labels)
#        total_test_loss += loss.item() * inputs.size(0)
#        total_test_samples += inputs.size(0)
#
#mse = total_test_loss / total_test_samples
#print(f'Mean Square Error on the test set: {mse:.4f}')
#
## Calcolo e stampa delle precisioni per il validation set
#accuracies_V, accuracies_P = test_accuracy(net, train_dataloader)
#print("Train set:")
#print(f'Velocity accuracy: {accuracies_V:.2f} %')
#print(f'Position accuracy: {accuracies_P:.2f} %')
#
## Calcolo e stampa delle precisioni per il validation set
#accuracies_V, accuracies_P = test_accuracy(net, val_dataloader)
#print("Validation set:")
#print(f'Velocity accuracy: {accuracies_V:.2f} %')
#print(f'Position accuracy: {accuracies_P:.2f} %')
#
## Calcolo e stampa delle precisioni per il test set
#accuracies_V, accuracies_P = test_accuracy(net, test_dataloader)
#print("Test set:")
#print(f'Velocity accuracy: {accuracies_V:.2f} %')
#print(f'Position accuracy: {accuracies_P:.2f} %')