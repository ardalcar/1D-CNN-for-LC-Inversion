import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from CNN_Pytorch import NeuralNetwork as NN, X,y
import matplotlib.pyplot as plt

# Creazione di un'istanza del modello
model = NN(input_channels=1,  kernel_size1=3, kernel_size2=3, kernel_size3=3, kernel_size4=3, hidden_units=64, output_units=6, batch_size=52)

# Definizione degli iperparametri da esplorare con la grid search
param_grid = {
    'input_channels': [1, 3],
    'kernel_size1': [2, 8],
    'kernel_size2': [2, 8],
    'kernel_size3': [2, 8],
    'kernel_size4': [2, 8],
    'hidden_units': [32, 64],
    'batch_size': [32, 52]
}

# Creazione di un'istanza del GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Esecuzione della grid search
grid_search.fit(X, y)

# Stampa dei risultati
print("Migliori parametri:", grid_search.best_params_)
print("Migliore punteggio:", grid_search.best_score_)