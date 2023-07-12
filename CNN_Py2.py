import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.conv1 = nn.Conv1d(1, 25, kernel_size=3).cuda() # input channel, filter size, kernel size
        self.pool = nn.MaxPool1d(kernel_size=2).cuda()       # kernel size, padding
        self.conv2 = nn.Conv1d(25,50,kernel_size=3).cuda()     # input channel, filter size, kernel size
        self.l1 = nn.Linear(29900, 25).cuda()       # input, hidden units
        self.l2 = nn.Linear(25, 10).cuda()        # input, hidden units
        self.l3 = nn.Linear(10, 6).cuda()          # input, hidden units
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
net = NeuralNetwork()
net.cuda()

# iperparametri
lr = 0.2       # learning rate
momentum = 0.001 # momentum
max_epoch = 20       # numero di epoche
batch_size = 10  # batch size

# ottimizzatori
criterion = nn.MSELoss().cuda()
#optimizer = optim.Adam(net.parameters(), lr)
optimizer = optim.SGD(net.parameters(), lr)

# Definizione dataloader per caricare i dati di addestramento
train_dataset = torch.utils.data.TensorDataset(inputs, labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        
# Ciclo di addestramento
for epoch in range(max_epoch):
    net.train()
    total_loss = 0

    for batch in train_dataloader:
        batch_inputs, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = net(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{max_epoch}], Loss: {avg_loss}")

# Salva il modello addestrato
torch.save(net.state_dict(), "modello_addestrato.pth")