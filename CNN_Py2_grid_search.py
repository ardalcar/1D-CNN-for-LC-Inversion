from CNN_Py2 import NeuralNetwork
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Database da file
X = np.load('X.npy')
y = np.load('y.npy')

# Divisione del dataset in addestramento e verifica in modo casuale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

# Conversione dei dati di input in tensori di PyTorch
inputs = torch.from_numpy(X_train).unsqueeze(1).float()
labels = torch.from_numpy(y_train).float()


# modello con skorch
net = NeuralNetRegressor(
    module=NeuralNetwork(),
    criterion = nn.MSELoss, 
    #optimizer = optim.Adam,
    #batch_size = 25,
    max_epochs = 10,
    #optimizer__lr = 0.001,
    verbose = False
    )



# definisce i parametri del grid search
param_grid = {
    'batch_size' : [10, 25, 50],
    'optimizer__lr' : [0.001, 0.01, 0.1, 0.2, 0.3],    #--> 0.001
    #'optimirer__momentum' : [0.0, 0.2, 0.4, 0.6, 0.8, 0.9] 
    'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta, #--> optim.Adam
                  optim.Adam, optim.Adamax, optim.NAdam]

}

grid = GridSearchCV(estimator=net, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(inputs,labels)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Salva il modello addestrato
torch.save(net.state_dict(), "modello_addestrato.pth")