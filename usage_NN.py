import CNN_Py2 as P2
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


modello_addestrato = P2.DataParallelModel()
modello_addestrato.load_state_dict(torch.load("modello_addestrato.pth"))

input_da_valutare = P2.inputs
input_da_valutare = input_da_valutare.to(P2.device(1))

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
    y_train, y_test = train_test_split(y, test_size=0.95, random_state=seed)
    print('y_true = ', y_train.shape)
else:
    y_train, y_test = train_test_split(y, test_size=0.95, random_state=seed)
    y_pred, y_test = train_test_split(y_pred, test_size=0.95, random_state=seed)

y_true, y_test_t = train_test_split(y_train, test_size=0.95, random_state=seed)
y_pred, y_test_p = train_test_split(y_pred, test_size=0.95, random_state=seed)

print('y_pred = ', y_pred.shape)
print('y_true = ', y_true.shape)


# Creazione dei grafici
for i in range(6):
    plt.figure()
    plt.plot(y_true[:, i], label='y_true')
    plt.plot(y_pred[:, i], label='y_pred')
    plt.xlabel('Sample')
    if i < 3:
        plt.ylabel('Magnitude (rad/s)')
        plt.title("Angular velocity")
    else:
        plt.ylabel('Value (rad)')
        plt.title('Euler Angle')

    plt.legend()
    nome_file=f"grafico_{i}.png"
    plt.savefig(nome_file)
    plt.clf()
    plt.close()

# Mostra i grafici
del input_da_valutare
del previsioni
plt.show()