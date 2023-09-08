import CNN_Py2 as P2
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


modello_addestrato = P2.NeuralNetwork()
modello_addestrato.load_state_dict(torch.load("modello_addestrato.pth"))

input_da_valutare = P2.inputs
input_da_valutare = input_da_valutare.to(P2.device)

modello_addestrato.eval()
previsioni = modello_addestrato(input_da_valutare)

y_pred = previsioni
y_pred = y_pred.cpu()
y_pred = y_pred.detach()
y_pred = y_pred.numpy()
print('y_pred = ', y_pred.shape)
# carico dataset true
with open('y2', 'rb') as file:
    y = pickle.load(file)

# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Divisione del dataset in addestramento e verifica in modo casuale
if P2.reduce:
    y_train, y_test = train_test_split(y, test_size=P2.r, random_state=seed)
    print('y_true = ', y_train.shape)
else:
    y_train, y_test = train_test_split(y, test_size=0.95, random_state=seed)
    y_pred, y_test = train_test_split(y_pred, test_size=0.95, random_state=seed)

y_true, y_test_t = train_test_split(y_train, test_size=0.95, random_state=seed)
y_pred, y_test_p = train_test_split(y_pred, test_size=0.95, random_state=seed)

print('y_pred = ', y_pred.shape)
print('y_true = ', y_true.shape)

# titoli e voci
title=[]
title.append('First')
title.append('Second')
title.append('Third')
title2=[]
title2.append('First')
title2.append('Second')
title2.append('Third')
angle=[]
angle.append('θ')
angle.append('φ')
angle.append('ψ')
vel_ang=[]
vel_ang.append('p')
vel_ang.append('q')
vel_ang.append('r')


# Creazione dei grafici
for i in range(6):
    plt.figure()
    plt.plot(y_true[:, i], label='y_true')
    plt.plot(y_pred[:, i], label='y_pred')
    plt.xlabel('Sample')
    if i < 3:
        plt.ylabel(f'{vel_ang.pop(0)} (rad/s)')
        plt.title(f'{title.pop(0)} Angular velocity')
    else:
        plt.ylabel(f'{angle.pop(0)} (rad)')
        plt.title(f'{title2.pop(0)} Euler Angle')

    plt.legend()
    nome_file=f"grafico_{i}.png"
    plt.savefig(nome_file)
    plt.clf()
    plt.close()