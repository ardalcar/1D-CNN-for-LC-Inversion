import pickle
import numpy as np

# Carica i dati dal file X.npy
datax = np.load('X2.npy')

with open('X2', 'wb') as f:
    pickle.dump(datax, f, pickle.HIGHEST_PROTOCOL)

# Carica i dati dal file y.npy
datay = np.load('y2.npy')

# Salva i dati pickled in un file .npy
with open('y2', 'wb') as file:
    pickle.dump(datay, file, pickle.HIGHEST_PROTOCOL)
