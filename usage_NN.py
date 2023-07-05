import numpy as np
from CNN_Pytorch import model

import torch
import matplotlib.pyplot as plt

# Carica il modello addestrato
#model = NN(input_channels_C1=1, filter_size_C1=1., kernel_size_C1=1, kernel_size_M1=1, padding_M1=1, 
#                      input_channels_C2=1, filter_size_C2=1, kernel_size_C2=1, kernel_size_M2=1, hidden_units=2400, 
#                      output_units=6, batch_size=2400)

# Carica i pesi del modello addestrato
model.load_state_dict(torch.load('modello_addestrato.pth'))
model.eval()

# Carica il dataset di input e output veri
X = np.load('X.npy')
y_true = np.load('y.npy')


# Conversione dei dati di input in tensori di PyTorch
inputs = torch.from_numpy(X).unsqueeze(1).float()

# Esegui le previsioni utilizzando il modello
with torch.no_grad():
    outputs = model(inputs)

# Confronta i risultati con i valori veri
y_pred = outputs.numpy()

# Stampa i risultati di previsione
#print("Risultati previsti:")
#print(y_pred)

#print("Risultati veri:")
#print(y_true)

# Creazione dei grafici
for i in range(6):
    plt.figure()
    plt.plot(y_true[:, i], label='y_true')
    plt.plot(y_pred[:, i], label='y_pred')
    plt.xlabel('Sample')
    plt.ylabel(f'Output {i+1}')
    plt.legend()

# Mostra i grafici
plt.show()