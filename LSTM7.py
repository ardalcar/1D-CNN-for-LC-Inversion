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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

################################## Neural Network ################################

class LSTMNet(nn.Module):
    def __init__(self, hidden_size, output_size=6): 
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=50, hidden_size=hidden_size, batch_first=True)  # input_size è 50 dopo la PCA
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(50)  # Aggiornato per riflettere la dimensione dell'input

    def forward(self, x):
        # Applicazione del dropout e della batch normalization
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)

        # LSTM
        output, (h, c) = self.lstm(x)
        out = h[-1]  # Ultimo layer dell'output dello stato nascosto
        out = out.view(out.size(0), -1)  # Modifica la forma se necessario
        out = self.fc(out)
        
        return out

# Definizione delle dimensioni degli strati
hidden_size = 100  # Dimensione dell'hidden layer LSTM
output_size = 6  # Dimensione dell'output

# Creazione dell'istanza della rete neurale
net = LSTMNet(hidden_size, output_size)
net.to(device)

# Stampa dell'architettura della rete
print(net)

# iperparametri
lr = 0.001        # learning rate
momentum = 0.001  # momentum
max_epoch = 200  # numero di epoche
batch_size = 128  # batch size
scaler = GradScaler()

criterion = nn.L1Loss().to(device)
#optimizer = optim.SGD(net.parameters(), lr)
optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=0.0001) # Regularizzazione L2 (Weight Decay)



##################################### carico dataset ##########################

# Funzione per applicare il windowing
def apply_windowing(data, window_size):
    windowed_data = []
    for sequence in data:
        if len(sequence) >= window_size:
            for i in range(0, len(sequence) - window_size + 1, window_size):
                windowed_data.append(sequence[i:i + window_size])
    return windowed_data

def MyDataLoader(X, y):
    window_size = 200
    pca = PCA(n_components=50)
    dbscan = DBSCAN(eps=0.5, min_samples=10)

    # Trasformazione dei dati
    windowed_curves, labels = [], []
    for sequence, label in zip(X, y):
        if len(sequence) >= window_size:
            for i in range(0, len(sequence) - window_size + 1, window_size):
                windowed_curves.append(sequence[i:i + window_size])
                labels.append(label)  # Aggiungi l'etichetta corrispondente

    # Padding e PCA
    padded_windows = pad_sequence([torch.tensor(window) for window in windowed_curves], batch_first=True, padding_value=0).numpy()
    X_reduced = pca.fit_transform(padded_windows.reshape(len(padded_windows), -1))

    # Rimozione degli outlier
    clusters = dbscan.fit_predict(X_reduced)
    non_outliers = X_reduced[clusters != -1]
    labels = np.array(labels)[clusters != -1]

    # Preparazione del DataLoader
    data_tensors = torch.tensor(non_outliers, dtype=torch.float32)
    label_tensors = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(data_tensors, label_tensors)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader, pca


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

def postprocess_sample(reconstructed_sample, pca, original_length, window_size=200):
    """
    Applica il processo inverso del preprocesso al campione ricostruito.

    :param reconstructed_sample: Il campione ricostruito dall'autoencoder.
    :param pca: L'oggetto PCA già addestrato.
    :param original_length: La lunghezza originale del campione prima del preprocesso.
    :param window_size: La dimensione della finestra utilizzata nel preprocesso.
    :return: Campione post-processato.
    """

    # Inversione della PCA
    sample_inverted_pca = pca.inverse_transform(reconstructed_sample)

    # Rimuovi il padding
    # Assumendo che il padding sia stato aggiunto alla fine di ciascuna finestra
    sample_no_padding = [window[:original_length] for window in sample_inverted_pca]

    # Ricostruisci la serie temporale dalle finestre
    # Questo dipende da come il windowing è stato applicato. Se non ci sono sovrapposizioni,
    # è possibile semplicemente concatenare le finestre.
    reconstructed_series = np.concatenate(sample_no_padding)

    return reconstructed_series

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

train_dataloader = MyDataLoader(X_train, y_train)
test_dataloader = MyDataLoader(X_test, y_test)
val_dataloader = MyDataLoader(X_val, y_val)

################################ Ciclo di addestramento ###############################################

writer = SummaryWriter('tensorboard/LSTM7')
loss_spann = []
loss_spann_val = []  # Per tenere traccia della loss sul validation set
gradient_spann = []

patience = 20  # Numero di epoche da attendere dopo l'ultimo miglioramento
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(max_epoch):
    # Training loop
    for i, (input, labels) in enumerate(train_dataloader):  
        input = input.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = net(input)
        outputs = outputs.squeeze(0) 
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()  
        loss.backward()

        # Stampa i gradienti
         # Registrazione dei gradienti per TensorBoard
        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                writer.add_scalar(f'Gradient/{name}', parameter.grad.norm().item(), epoch)
                gradient_spann.append(parameter.grad.norm().item())

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()

    # Calcolo della loss sul validation set
    with torch.no_grad():
        total_val_loss = 0
        total_samples = 0
        for input_val, labels_val in val_dataloader:
            input_val = input_val.to(device)
            labels_val = labels_val.to(device)

            outputs_val = net(input_val)
            loss_val = criterion(outputs_val, labels_val)

            total_val_loss += loss_val.item() * len(labels_val)
            total_samples += len(labels_val)

        average_val_loss = total_val_loss / total_samples
        loss_spann_val.append(average_val_loss)

    writer.add_scalar('Loss/Train', loss.item(), epoch)
    writer.add_scalar('Loss/Validation', average_val_loss, epoch)

    print(f'Epoch [{epoch+1}/{max_epoch}] Loss: {loss.item():.4f} Loss validation: {average_val_loss:.4f}')
    loss_spann.append(loss.item())

    # Controllo per l'early stopping
    if average_val_loss < best_loss:
        best_loss = average_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print("Early stopping triggered")
        break


# Salva il modello addestrato
model_save_path = 'models/LSTM7.pth'
torch.save(net.state_dict(), model_save_path)

# Salva i log delle loss
with open('grad_spannLSTM7.txt', 'w') as file:
    for valore in gradient_spann:
        file.write(str(valore) + '\n')

# Salva i log delle loss
with open('losses/loss_spannLSTM7.txt', 'w') as file:
    for valore in loss_spann:
        file.write(str(valore) + '\n')

with open('losses/loss_spannLSTM7_val.txt', 'w') as file:
    for valore in loss_spann_val:
        file.write(str(valore) + '\n')


################################ Test Modello #############################################

# Carico modello
net=LSTMNet(hidden_size=hidden_size, output_size=output_size)
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Test set aaoooooooo

dataiter = iter(test_dataloader)
#inputs, labels = next(dataiter)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    loss = 0
    for input, labels in test_dataloader:
        input = input.to(device)
        labels = labels.to(device)
        outputs = net(input)
        loss += criterion(outputs, labels).item()


    mse = loss / len(test_dataloader)
    print(f'Mean Square Error on the test set: {mse} %')


# Test 
def test_accuracy(net, test_dataloader):
    net.eval()  # Imposta la rete in modalità valutazione
    predicted = []
    reals = []
    with torch.no_grad():
        for inputs, real in test_dataloader:  # Rimuovi la gestione delle lunghezze
            inputs, real = inputs.to(device), real.to(device)
            output = net(inputs)  # Rimuovi il parametro lengths
            predicted.append(output)
            reals.append(real)

        predicted = torch.cat([p for p in predicted], dim=0)
        reals = torch.cat([r for r in reals], dim=0)

    # get the accuracy for all value
    errors = reals - predicted
    errors= torch.Tensor.cpu(errors)
    errors = torch.abs(errors)

    # get best fitted curve
    med_errors = torch.sum(errors, axis=1)
    min_error = torch.min(med_errors)
    index_min = torch.argmin(med_errors)
    print("Errore minimo: ",min_error)
    print(f'Assetto originale: {reals[index_min,:]}')
    print(f'Assetto trovato: {predicted[index_min,:]}')

    tollerance_velocity=0.0001*10000
    tollerance_position=0.0174533

    # error like True or False
    errors_V = errors[:,0:3]
    errors_P = errors[:,3:6]
    boolean_eV = errors_V <= tollerance_velocity
    boolean_eP = errors_P <= tollerance_position

    float_tensor_V = boolean_eV.float()
    float_tensor_P = boolean_eP.float()


    accuracies_V = float_tensor_V.mean(dim=0)*100
    accuracies_P = float_tensor_P.mean(dim=0)*100
    accuracies_V=torch.Tensor.numpy(accuracies_V)
    accuracies_P=torch.Tensor.numpy(accuracies_P)

    return accuracies_V, accuracies_P
# Print accuracies

print()
########
accuracies_V, accuracies_P = test_accuracy(net,train_dataloader)
print('Train set:')
print()
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')

print()
########
accuracies_V, accuracies_P = test_accuracy(net,val_dataloader)
print('Validation set:')
print()
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')

accuracies_V, accuracies_P = test_accuracy(net,test_dataloader)
print('Testset:')
print()
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')
