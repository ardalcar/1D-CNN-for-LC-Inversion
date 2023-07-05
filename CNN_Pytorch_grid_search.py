import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
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

# Modello della rete neurale
class NeuralNetwork(nn.Module):
    def __init__(self, input_channels_C1, filter_size_C1, kernel_size_C1, kernel_size_M1, 
                 padding_M1, input_channels_C2, filter_size_C2, kernel_size_C2, kernel_size_M2, 
                 hidden_units, output_units):
       # super(NeuralNetwork, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels_C1, int(filter_size_C1), int(kernel_size_C1))
        self.maxpool1 = nn.MaxPool1d(kernel_size_M1, padding_M1)
        self.conv2 = nn.Conv1d(input_channels_C2,filter_size_C2, kernel_size_C2)
        self.maxpool2 = nn.MaxPool1d(kernel_size_M2)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2400, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_units),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# modello con skorch
model = NeuralNetRegressor(
    NeuralNetwork(input_channels_C1=1, filter_size_C1=1., kernel_size_C1=1, kernel_size_M1=1, padding_M1=1, 
                      input_channels_C2=1, filter_size_C2=1, kernel_size_C2=1, kernel_size_M2=1, hidden_units=2400, 
                      output_units=6).to(device),
    criterion = nn.MSELoss,optimizer = optim.Adam,
    verbose = False
    )
print(model)


# definisce i parametri del grid search
param_grid = {
    'batch_size': [10, 20, 40, 60, 80, 100],
    'max_epochs': [10, 50, 100]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(inputs,labels)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Definisci il numero di epoche, il learning rate e altre iperparametri
#num_epochs = 10
#learning_rate = 0.001
#batch_size = 2400


# Creazione di un'istanza del modello
#model = NeuralNetwork(input_channels_C1=1, filter_size_C1=1., kernel_size_C1=1, kernel_size_M1=1, padding_M1=1, 
#                      input_channels_C2=1, filter_size_C2=1, kernel_size_C2=1, kernel_size_M2=1, hidden_units=2400, 
#                      output_units=6).to(device)
#print(model)

# Definizione ottimizzatore e la loss
#optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#criterion = nn.MSELoss()

# Definisci il dataloader per caricare i dati di addestramento
#train_dataset = torch.utils.data.TensorDataset(inputs, labels)
#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Ciclo di addestramento
#for epoch in range(num_epochs):
#    model.train()
#    total_loss = 0

#    for batch in train_dataloader:
#        batch_inputs, batch_labels = batch
#        batch_inputs = batch_inputs.to(device)
#        batch_labels = batch_labels.to(device)

#        optimizer.zero_grad()
#        outputs = model(batch_inputs)
#        loss = criterion(outputs, batch_labels)
#        loss.backward()
#        optimizer.step()

#        total_loss += loss.item()

#    avg_loss = total_loss / len(train_dataloader)
#    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

# Salva il modello addestrato
#torch.save(model.state_dict(), "modello_addestrato.pth")