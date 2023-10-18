import torch
import torch.optim as optim
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from Rete_Neurale2 import NeuralNetwork2
from skorch import NeuralNetRegressor
import torch.nn as nn


device = (
    f"cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

################## Neural Network ################


############### carico dataset ##########################

with open("./data/X2", 'rb') as file:
    X = pickle.load(file)

with open("./data/y2", 'rb') as file:
    y = pickle.load(file)

# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


z=len(X)
print(f'Il dataset contiene {z} samples')
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
z2=len(X_train)
z3=len(X_test)
print(f'Il trainset contiene {z2} samples')
print(f'Il testset contiene {z3} samples')
inputs = torch.from_numpy(X_train).unsqueeze(1).float()
labels = torch.from_numpy(y_train).float()

train_dataset = TensorDataset(inputs, labels)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

################################ Grid Seach ####################################


# Definisci la funzione per la creazione del modello
model = NeuralNetRegressor(
    module=NeuralNetwork2(
        filter_size1=32, 
        kernel_size1=3, 
        filter_size2=32, 
        kernel_size2=3),
        criterion = nn.MSELoss,
        optimizer = optim.Adam,
        batch_size = 25,
        max_epoch = 10,
        optimizer_lr = 0.001,
        verbose = False
)


# Definisci la griglia di iperparametri da esplorare
param_grid = {
    #'filter_size1': [16, 32],
    #'kernel_size1': [3, 5],
    #'filter_size2': [32, 64],
    #'kernel_size2': [3, 5],
    'batch_size' : [25, 30, 35],
    'optimizer__lr' : [0.001, 0.01, 0.1, 0.2, 0.3],
    'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta, #--> optim.Adam
                  optim.Adam, optim.Adamax, optim.NAdam]
}

# Definisci il regressore con la grid search
regressor = GridSearchCV(estimator=model, 
                         param_grid=param_grid, 
                         n_jobs=-1,
                         scoring='neg_mean_squared_error', 
                         cv=3)

# Esegui la grid search
grid_result = regressor.fit(X_train, y_train)

# Ottieni il modello con i migliori iperparametri
best_model = regressor.best_estimator_

# Valuta il modello sui dati di test
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
