import torch
import torch.optim as optim
import torch.nn as nn
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


############################# Carico Dataset ###################################

with open("./dataCNN/X2", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y2", 'rb') as file:
    y = pickle.load(file)


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Variabile per controllare se eseguire l'addestramento o meno


z=len(X)
print(f'Il dataset contiene {z} samples')

r=0.25

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=seed)
z2=len(X_train)
z3=len(X_test)
print(f'Il trainset contiene {z2} samples')
print(f'Il testset contiene {z3} samples')
inputs = torch.from_numpy(X_train).unsqueeze(1).float()
labels = torch.from_numpy(y_train).float()

batch_size = 20
train_dataset = TensorDataset(inputs, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 


################## Neural Network ################
from Rete_Neurale2 import NeuralNetwork2

# iperparametri
lr = 0.2          # learning rate
momentum = 0.001  # momentum
max_epoch = 1   # numero di epoche
batch_size = 5   # batch size
scaler = GradScaler()

# ottimizzatori
criterion = nn.MSELoss().to(device)
#optimizer = optim.Adam(net.parameters(), lr)
print('a=0')
a=0
b=0
for l in range(1,5000):
    for k in range(1,5):
        for j in range(1,5):
            for i in range(1,5):
                a+=1
                print(f'a={a} b={b}')
                try:
                    kernel_size1=i
                    kernel_size2=j
                    kernel_size3=k
                    initial_step=l
                    net = NeuralNetwork2(kernel_size1, kernel_size2, kernel_size3, initial_step)
                    net.to(device)
 ########################## Ciclo di addestramento ####################
                    optimizer = optim.SGD(net.parameters(), lr)
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
                    #with open('loss_spann.txt', 'w') as file:
                    #    for valore in loss_spann:
                    #        file.write(str(valore) +'\n')
                    with open('parametri_funzionanti.txt','a') as file:
                        file.write(f'kernel_size1={i} kernel_size2={j} kernel_size3={k} initial_step={l} funziona \n')
                    print(f'kernel_size1={i} kernel_size2={j} kernel_size3={k} initial_step={l} funziona')
                    b-=1
                except:
                    b+=1
                    continue
                    #print(f'ffs1={i} ks1={j} fs2={k} ks2={l} ks3={n} ins={m} non funziona')

#import multiprocessing
#import signal
#import sys
#
#def signal_handler(signal, frame):
#    # Questa funzione verrà chiamata quando viene premuto Ctrl+C
#    raise KeyboardInterrupt("Interruzione manuale")
#
## Funzione che addestra il modello e restituisce il risultato
#def train_and_evaluate(args):
#    ffs1, ks1, fs2, ks2, ks3, ins = args
#
#    try:
#        filter_size1 = ffs1
#        kernel_size1 = ks1
#        filter_size2 = fs2
#        kernel_size2 = ks2
#        kernel_size3 = ks3
#        initial_step = ins
#
#        net = NeuralNetwork2(filter_size1, kernel_size1, filter_size2, kernel_size2, kernel_size3, initial_step)
#        net.to(device)
#        
#        optimizer = optim.SGD(net.parameters(), lr)
#        loss_spann = []
#        
#        for epoch in range(max_epoch):
#            net.train()
#            total_loss = 0
#            
#            for batch in train_dataloader:
#                batch_inputs, batch_labels = batch
#                batch_inputs = batch_inputs.to(device)
#                batch_labels = batch_labels.to(device)
#
#                optimizer.zero_grad()
#
#                with autocast():
#                    outputs = net(batch_inputs)
#                    loss = criterion(outputs, batch_labels)
#                scaler.scale(loss).backward()
#                scaler.step(optimizer)
#                scaler.update()
#
#                total_loss += loss.item()
#
#            avg_loss = total_loss / len(train_dataloader)
#            loss_spann.append(avg_loss)
#        
#        with open('parametri_funzionanti.txt', 'a') as file:
#            file.write(f'ffs1={ffs1} ks1={ks1} fs2={fs2} ks2={ks2} ks3={ks3} ins={ins} loss={avg_loss} funziona\n')
#
#        print(f'ffs1={ffs1} ks1={ks1} fs2={fs2} ks2={ks2} ks3={ks3} ins={ins} funziona')
#   
#    except KeyboardInterrupt as e:
#        print("Programma interrotto manualmente:", e)
#        sys.exit()
#
#    except:
#        pass
#
#if __name__ == '__main__':
#    # Lista di tuple con gli argomenti da passare alla funzione
#    args_list = [(i, j, k, l, m, n) for n in range(1, 5000) for m in range(1, 5) for l in range(1, 5) for k in range(1, 5) for j in range(1, 5) for i in range(1, 5)]
#    
#    # Utilizza multiprocessing per eseguire le funzioni in parallelo
#    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#        pool.map(train_and_evaluate, args_list)


