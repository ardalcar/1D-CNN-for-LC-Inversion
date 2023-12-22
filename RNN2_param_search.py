import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

device = (
    f"cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

################################## Neural Network ################################

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Feed-forward layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        # Output layer
        self.fc3 = nn.Linear(32, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Select the last time step output from the LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Feed-forward layers with ReLU activation
        out = self.relu1(self.fc1(lstm_out))
        out = self.relu2(self.fc2(out))
        
        # Output layer with tanh activation
        out = self.tanh(self.fc3(out))
        
        return out


# iperparametri
lr = 0.2          # learning rate
momentum = 0.001  # momentum
max_epoch = 1000   # numero di epoche
batch_size = 20   # batch size
scaler = GradScaler()


##################################### carico dataset ##########################

with open("./dataCNN/X2", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y2", 'rb') as file:
    y = pickle.load(file)


                       
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

r=0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=seed)

# Train set
inputs = torch.from_numpy(X_train).unsqueeze(1).float()
labels = torch.from_numpy(y_train).float()

train_dataset = TensorDataset(inputs, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Test set
inputs = torch.from_numpy(X_test).unsqueeze(1).float()
labels = torch.from_numpy(y_test).float()

test_dataset = TensorDataset(inputs, labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



################################ Ciclo di addestramento ###############################################

def addestramento(index):
    # Definizione delle dimensioni degli strati
    input_size = 2400  # Dimensione dell'input
    hidden_size = index  # Dimensione dell'hidden layer LSTM
    output_size = 6  # Dimensione dell'output

    # Creazione dell'istanza della rete neurale
    net = RNN(input_size, hidden_size, output_size).to(device)

    # Stampa dell'architettura della rete
    print(net)

    # ottimizzatori
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr)

    # Train the model
    n_total_steps = len(train_dataloader)
    for epoch in range(max_epoch):
        for i, (images, labels) in enumerate(train_dataloader):  
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 28, 28]
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calcolo della loss sul test set
        with torch.no_grad():
            total_test_loss = 0
            total_samples = 0
            for images_test, labels_test in test_dataloader:
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = net(images_test)
                loss_test = criterion(outputs_test, labels_test)

                total_test_loss += loss_test.item() * len(labels_test)
                total_samples += len(labels_test)

            average_test_loss = total_test_loss / total_samples


        #print (f'Epoch [{epoch+1}/{max_epoch}] Loss: {loss.item():.4f} Loss test: {loss_test.item():.4f}')

    mse = loss / len(test_dataloader)
    print(f'Mean Square Error on the test set: {mse} %')
    return loss, mse, index


#for i in range(1,500):
def main():
    index=500
    loss, mse, indey = [addestramento(i) for i in range(1,index)]
    print(f'hidden size: {indey}, loss: {loss:.6f}, mse: {mse:.6f}')



if __name__ == "__main__":
    main()