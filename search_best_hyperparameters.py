import numpy as np
from sklearn.model_selection import GridSearchCV
from CNN_Pytorch import NeuralNetwork as NN, X,y
import matplotlib.pyplot as plt




# Creazione di un'istanza del modello
model = NN(input_channels_C1=1, filter_size_C1=1., kernel_size_C1=1, kernel_size_M1=1, padding_M1=1, 
                      input_channels_C2=1, filter_size_C2=1, kernel_size_C2=1, kernel_size_M2=1, hidden_units=2400, 
                      output_units=6)
# Definizione degli iperparametri da esplorare con la grid search

#input_channels_C1=1, filter_size_C1=1, kernel_size_C1=1, kernel_size_M1=1, padding_M1=1, 
#input_channels_C2=1, filter_size_C2=1, kernel_size_C2=1, kernel_size_M2=1, hidden_units=2400, 
#output_units=6, batch_size=2400
param_grid = {
    'input_channels_C1':            [1, 3, 5],
    'filter_size_C1':               [1, 3, 5],
    'kernel_size_C1':               [1, 3, 5],
    'kernel_size_M1':               [1, 3, 5],
    'padding_M1':                   [1, 3, 5],
    'input_channels_C2':            [1, 3, 5],
    'filter_size_C2':               [1, 3, 5],
    'kernel_size_C2':               [1, 3, 5],
    'kernel_size_M2':               [1, 3, 5],
    'hidden_units':                 [2400],
    'output_units':[6],
    'batch_size': [32, 52]
}

# Impostazione errore per evitare che si blocchi su iperparametri non funzionanti
#grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', error_score=-1)

# Creazione di un'istanza del GridSearchCV
#grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Impostazione errore in NaN per evitare che si blocchi su iperparametri non funzionanti
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', error_score=np.nan)


# Esecuzione della grid search
grid_search.fit(X, y)

# Stampa dei risultati
print("Migliori parametri:", grid_search.best_params_)
print("Migliore punteggio:", grid_search.best_score_)