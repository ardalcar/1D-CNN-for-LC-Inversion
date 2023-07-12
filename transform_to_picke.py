import pickle
import numpy as np

# Carica i dati dal file X.npy
datax = np.load('X.npy')
datay = np.load('y.npy')

# Salva i dati come pickle
with open('X.pickle', 'wb') as file:
    pickle.dump(datax, file)

with open('y.pickle', 'wb') as file:
    pickle.dump(datay, file)
