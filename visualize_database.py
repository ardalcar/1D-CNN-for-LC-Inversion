import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Carica il database da file
with open('dataCNN/X3','rb') as file:
    X_train = pickle.load(file)
with open('dataCNN/y3','rb') as file:
    y_train = pickle.load(file)

# Seleziona un dato di addestramento specifico da visualizzare
dato_idx = 1

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
