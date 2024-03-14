import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Carica il database da file
folder = sys.argv[1]
num = sys. argv[2]
X_data = 'X' + num
y_data = 'y' + num
pathX = os.path.join('.', folder, X_data)
pathy = os.path.join('.', folder, y_data)
with open(pathX,'rb') as file:
    X_train = pickle.load(file)
with open(pathy,'rb') as file:
    y_train = pickle.load(file)

# Seleziona un dato di addestramento specifico da visualizzare
dato_idx = sys. argv[3]
dato_idx = np.int64(dato_idx)

# Estrai l'input e l'output corrispondenti al dato selezionato
input_dato = X_train[dato_idx]
output_dato = y_train[dato_idx]

# Grafico dell'input
plt.subplot(2, 1, 1)
plt.plot(input_dato)
plt.title('Input Dato di Addestramento')

# Grafico dell'output
plt.subplot(2, 1, 2)
plt.plot(output_dato)
plt.title('Output Dato di Addestramento')

# Mostra i grafici a schermo
plt.tight_layout()
plt.savefig('fname')
plt.show()
