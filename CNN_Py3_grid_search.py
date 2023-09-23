import torch
import io
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

######################## Load dataset #####################################################

def load_data(data_dir="./dataCNN"):
    # carico dataset
    stringa1=data_dir+"/X2"
    with open(stringa1, 'rb') as file:
        X = pickle.load(file)

    stringa2=data_dir+"/y2"
    with open(stringa2, 'rb') as file:
        y = pickle.load(file)

    # Seme per la generazione dei numeri casuali
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    r=0.25

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=seed)
    z2=len(X_train)
    z3=len(X_test)
    print(f'Il trainset contiene {z2} samples')
    print(f'Il testset contiene {z3} samples')

    X_train = torch.from_numpy(X_train).unsqueeze(1).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).unsqueeze(1).float()
    y_test = torch.from_numpy(y_test).float()

    # Crea una lista vuota per contenere le coppie di input e label
    trainset = []
    # Itera attraverso gli elementi di X e y
    for i in range(len(X_train)):
        input_sample = X[i]  # Prendi il campione i-esimo da X
        label_sample = y[i]  # Prendi il corrispondente campione i-esimo da y
        trainset.append((input_sample, label_sample))  # Aggiungi la coppia alla lista

    testset = []
    # Itera attraverso gli elementi di X e y
    for i in range(len(X_test)):
        input_sample = X[i]  # Prendi il campione i-esimo da X
        label_sample = y[i]  # Prendi il corrispondente campione i-esimo da y
        testset.append((input_sample, label_sample))  # Aggiungi la coppia alla lista


    return trainset, testset

########################## Neural Network ###################################################
class NeuralNetwork(nn.Module):
    def __init__(self,l1=120,l2=84):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 25, 3) # input channel, filter size, kernel size
        self.pool = nn.MaxPool1d(2)     # kernel size, padding
        self.conv2 = nn.Conv1d(25, 50, 3) # input channel, filter size, kernel size
        self.l1 = nn.Linear(29900, l1)              # input, hidden units
        self.l2 = nn.Linear(l1, l2)                 # input, hidden units
        self.l3 = nn.Linear(l2, 6)                  # input, hidden units
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
########################## Train Code ############################

def train_cifar(config, data_dir=None):
    net = NeuralNetwork(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.001)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total},
            checkpoint=checkpoint,
        )
    print("Finished Training")

############################## Test set accuracy #####################

def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    reals=[]
    with torch.no_grad():
        for data in testloader:
            inputs, real = data
            inputs, real = data[0].to(device), data[1].to(device)
            predicted = net(inputs.to(device))
            
            error= reals - predicted
            tollerance_velocity=0.01
            tollerance_position=1
            errors_V = error[0:3]
            errors_P = error[3:6]
            boolean_eV = errors_V <= tollerance_velocity
            boolean_eP = errors_P <= tollerance_position

            float_tensor_V = boolean_eV.float()
            float_tensor_P = boolean_eP.float()

            accuracies_V = float_tensor_V.sum()
            accuracies_P = float_tensor_P.sum()
            correct+=accuracies_P+accuracies_V
            total+=6

    return correct / total

################## MAIN ####################################à

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./dataCNN")
    load_data(data_dir)
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([1, 2, 4, 8, 16, 20]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = NeuralNetwork(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)