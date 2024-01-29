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
import math


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

################################## Neural Network ################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, num_heads, num_layers, hidden_size, embed_dim):
        super(TransformerModel, self).__init__()
        self.input_size = 1
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = 6

        self.embedding = nn.Linear(self.input_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)  # Inizializzazione del Positional Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, self.output_size)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Reshaping the input for Transformer
        transformer_output = self.transformer_encoder(src)
        output = self.fc_out(transformer_output[-1])
        return output



# Parametri del modello (esempio)

num_heads = 32   # Numero di testine nel Transformer
num_layers = 6  # Numero di layer nel Transformer
hidden_size = 1024  # Dimensione dello strato nascosto
embed_dim = hidden_size

# Creazione del modello
net = TransformerModel(num_heads, num_layers, hidden_size, embed_dim).to(device)

# Definire criterio di loss e ottimizzatore
criterion = nn.MSELoss()  # o un altro loss appropriato per la regressione
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Stampa dell'architettura della rete
print(net)

# iperparametri
lr = 0.001        # learning rate
momentum = 0.001  # momentum
max_epoch = 2000  # numero di epoche
batch_size = 128  # batch size
scaler = GradScaler()



##################################### carico dataset ##########################

with open("./dataCNN/X41", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y41", 'rb') as file:
    y = pickle.load(file)

with open("./dataCNN/X41", 'rb') as file:
    X1 = pickle.load(file)

with open("./dataCNN/y41", 'rb') as file:
    y1 = pickle.load(file)
    
                       
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
X_train=X
y_train=y

## Train set
inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_train], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_train).float()

lengths_train = [len(seq) for seq in X_train]
lengths_train_tensor = torch.LongTensor(lengths_train)

train_dataset = TensorDataset(inputs, labels, lengths_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

## Validation test
inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_val], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_val).float()

lengths_val = [len(seq) for seq in X_val]
lengths_val_tensor = torch.LongTensor(lengths_val)

val_dataset = TensorDataset(inputs, labels, lengths_val_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


## Test set
lengths_test = [len(seq) for seq in X_test]
lengths_test_tensor = torch.LongTensor(lengths_test)

inputs = pad_sequence([torch.tensor(seq).unsqueeze(-1) for seq in X_test], batch_first=True, padding_value=0)
labels = torch.from_numpy(y_test).float()

test_dataset = TensorDataset(inputs, labels, lengths_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

################################ Ciclo di addestramento ###############################################

writer = SummaryWriter('tensorboard/Transformer')
loss_spann = []
loss_spann_val = []  # Per tenere traccia della loss sul validation set

patience = 200  # Numero di epoche da attendere dopo l'ultimo miglioramento
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(max_epoch):
    # Training loop
    for i, (input, labels, lengths) in enumerate(train_dataloader):  
        input = input.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = net(input)#, lengths) 
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()

    # Calcolo della loss sul validation set
    with torch.no_grad():
        total_val_loss = 0
        total_samples = 0
        for images_val, labels_val, lengths_val in val_dataloader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            outputs_val = net(images_val)#, lengths_val)
            loss_val = criterion(outputs_val, labels_val)

            total_val_loss += loss_val.item() * len(labels_val)
            total_samples += len(labels_val)

        average_val_loss = total_val_loss / total_samples
        loss_spann_val.append(average_val_loss)

    writer.add_scalar('Loss/Train', loss, epoch)
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

writer.close()

# Salva il modello addestrato
model_save_path = 'Transformer5.pth'
torch.save(net.state_dict(), model_save_path)

# Salva i log delle loss
with open('loss_spannTransformer5.txt', 'w') as file:
    for valore in loss_spann:
        file.write(str(valore) + '\n')

with open('loss_spannTransformer5_val.txt', 'w') as file:
    for valore in loss_spann_val:
        file.write(str(valore) + '\n')


################################ Test Modello #############################################

# Carico modello
net=TransformerModel(num_heads, num_layers, hidden_size, embed_dim)
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Test set

dataiter = iter(test_dataloader)
#inputs, labels = next(dataiter)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    loss = 0
    for input, labels, lengths in test_dataloader:
        input = input.to(device)
        labels = labels.to(device)
        outputs = net(input)#, lengths)
        loss += criterion(outputs, labels).item()


    mse = loss / len(test_dataloader)
    print(f'Mean Square Error on the test set: {mse} %')


# Test 
def test_accuracy(net, test_dataloader=test_dataloader):
    net.eval()
    predicted=[]
    reals=[]
    with torch.no_grad():
        for data in test_dataloader:
            inputs, real, lengths = data
            inputs, real = inputs.to(device), real.to(device)
            output = net(inputs)#, data[2])
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

    tollerance_velocity=0.0001
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

accuracies_V, accuracies_P = test_accuracy(net,test_dataloader)
print('testset:')
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')

print()
########
accuracies_V, accuracies_P = test_accuracy(net,train_dataloader)
print('trainset:')
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')

print()
########
accuracies_V, accuracies_P = test_accuracy(net,val_dataloader)
print('validationset:')
for j in 0, 1, 2: 
    print(f'Velocity accuracy {j+1}: {accuracies_V[j]: .2f} %')

print()
for i in 0, 1, 2:
    print(f'Position accuracy {i+1}: {accuracies_P[i]: .2f} %')
