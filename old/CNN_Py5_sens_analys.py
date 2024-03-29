import torch
import io
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

device = (
    f"cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

######################## Neural Network #####################
from Rete_Neurale2 import NeuralNetwork2


kernel_size1=3
kernel_size2=4
kernel_size3=4
initial_step=149
net = NeuralNetwork2(kernel_size1=kernel_size1, 
                     kernel_size2=kernel_size2, 
                     kernel_size3=kernel_size3, 
                     initial_step=initial_step)
net.to(device)

# iperparametri
lr = 0.2          # learning rate
momentum = 0.001  # momentum
max_epoch = 200   # numero di epoche
batch_size = 20   # batch size
scaler = GradScaler()

print(net)

# ottimizzatori
if torch.cuda.is_available():
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()
#optimizer = optim.Adam(net.parameters(), lr)
optimizer = optim.SGD(net.parameters(), lr)


######## carico dataset ##########################
z = input('whitch subdataset use? [1,2,3]: ')
while z not in ['1', '2', '3']:
    print('input non valido, inserire solo 1, 2 o 3')
    z = input('whitch subdataset use? [1,2,3]: ')
    

# Carica il dizionario dei sotto-dataset
with open('./dataCNN/sub_datasets', 'rb') as file:
    sub_datasets = pickle.load(file)
z = eval(z)
# Carica il secondo sotto-dataset
sub_dataset = sub_datasets[z]

# Ora puoi accedere ai dati come segue
X = np.array(sub_dataset['X2'])
y = np.array(sub_dataset['y2'])

                       
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Variabile per controllare se eseguire l'addestramento o meno
train_model = input('Eseguire addestramento? [Yes/No] ').lower()

while train_model not in ['y', 'n', 'yes', 'no']:
    print("Input non valido. inserire 'y', 'n', 'yes' o 'no'.")
    train_model = input("Eseguire addestramento? [Yes/No] ").lower()

if train_model == 'y' or train_model == 'yes':
    train_model = True
elif train_model == 'n' or train_model == 'no':
    train_model = False


if train_model:
# Riduzione del dataset

    z=len(X)
    print(f'Il dataset contiene {z} samples')
    
    r_input=input("Inserire la quota parte del trainset: [1:99] ")
    if not r_input:
        r=0.25
    else:
        try:
            r2 = float(r_input)
            r = 100-r2
            r = r/100
            print(f'Trainset utilizzato: {r}%')
        except ValueError:
            print('Input non valido. Inserire un numero valido in formato float.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=seed)
    z2=len(X_train)
    z3=len(X_test)
    print(f'Il trainset contiene {z2} samples')
    print(f'Il testset contiene {z3} samples')
    inputs = torch.from_numpy(X_train).unsqueeze(1).float()
    labels = torch.from_numpy(y_train).float()
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    inputs = torch.from_numpy(X_train).unsqueeze(1).float()
    labels = torch.from_numpy(y_train).float()

train_dataset = TensorDataset(inputs, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
# Ciclo di addestramento
if train_model:
    loss_spann = []
    for epoch in range(max_epoch):
        net.train()
        total_loss = 0

        for batch in train_dataloader:
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            
                
            optimizer.zero_grad()

            with autocast():
                outputs = net(batch_inputs)
                loss = criterion(outputs, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{max_epoch}], Loss: {avg_loss}")

        loss_spann.append(avg_loss)
    
    with open('loss_spann.txt', 'w') as file:
        for valore in loss_spann:
            file.write(str(valore) +'\n')


    # Salva il modello addestrato
    model_save_path = 'modello_addestrato_sub.pth'
    torch.save(net.state_dict(),model_save_path)
else:
    model_save_path = 'modello_addestrato_sub.pth'

###########################################################


# Carico modello
net=NeuralNetwork2(kernel_size1=kernel_size1,  
                   kernel_size2=kernel_size2, 
                   kernel_size3=kernel_size3, 
                   initial_step=initial_step)
net.to(device)
net.load_state_dict(torch.load(model_save_path))

# Test set
inputs = torch.from_numpy(X_test).unsqueeze(1).float()
labels = torch.from_numpy(y_test).float()

test_dataset = TensorDataset(inputs, labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

dataiter = iter(test_dataloader)
#inputs, labels = next(dataiter)

# Test 
def test_accuracy(net, test_dataloader=test_dataloader):

    with torch.no_grad():
        predicted=[]
        reals=[]
        for data in test_dataloader:
            inputs, real = data[0].to(device), data[1].to(device)
            predict = net(inputs.to(device))
            predicted.append(predict)
            reals.append(real)

    reals = torch.cat(reals, dim=0)
    predicted = torch.cat(predicted, dim=0)

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
    tollerance_position=1

    # error like True or False
    num_row, num_col = errors.size() 
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



