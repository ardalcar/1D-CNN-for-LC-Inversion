import pickle
import numpy as np

# Carica i dati dal file X.npy
datax = np.load('X.npy')

# Trasforma i dati in formato pickled
datax_pickled = pickle.dumps(datax)

# Salva i dati pickled in un file .npy
with open('X.pickle.npy', 'wb') as file:
    np.save(file, datax_pickled)


# Carica i dati dal file y.npy
datay = np.load('y.npy', allow_pickle=True)

# Trasforma i dati in formato pickled
datay_pickled = pickle.dumps(datay)

# Salva i dati pickled in un file .npy
with open('y.pickle.npy', 'wb') as file:
    np.save(file, datay_pickled)
