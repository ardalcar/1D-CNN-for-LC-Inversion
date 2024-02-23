import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def truncate_to_shortest_and_convert_to_array(light_curves):
    # Trova la lunghezza della curva più corta
    min_length = min(len(curve) for curve in light_curves)

    # Tronca tutte le curve alla lunghezza della curva più corta e le converte in un array
    truncated_curves = [curve[:min_length] for curve in light_curves]
    array_curves = np.array(truncated_curves)

    return array_curves

def normalize_array(Input, max, min):
    norm_arr = (Input - min) / (max - min)
    return norm_arr

def normalize_y(y, max_angle=1.5, min_angle=-1.5, max_vel=0.0002, min_vel=-0.0002):
    y_angle=y[:,-3:]
    y_vel=y[:,:3]
    y_norm_angle=normalize_array(y_angle, max_angle, min_angle)
    y_norm_vel=normalize_array(y_vel, max_vel, min_vel)
    y_norm=np.concatenate((y_norm_vel, y_norm_angle), axis=1)
    return y_norm

def denormalize_array(norm_arr, max, min):
    input_arr = norm_arr * (max - min) + min
    return input_arr

def denormalize_y(y_norm, max_angle=1.5, min_angle=-1.5, max_vel=0.0002, min_vel=-0.0002):
    if len(y_norm.shape) == 1:
        y_norm = y_norm[np.newaxis, :]
        a=True
    else:
        a=False

    y_norm_vel = y_norm[:, :3]
    y_norm_angle = y_norm[:, -3:]
    
    y_vel = denormalize_array(y_norm_vel, max_vel, min_vel)
    y_angle = denormalize_array(y_norm_angle, max_angle, min_angle)

    y = np.concatenate((y_vel, y_angle), axis=1)
    if a:
        y = y.squeeze(0)

    print(y)
    return y

# carico dataset 
def MyDataLoader(X, y,  batch_size):
    # Converti in tensori
    data_tensors = torch.tensor(X, dtype=torch.float32)
    label_tensors = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(data_tensors, label_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return  dataloader


################################### main ###################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

with open("./dataCNN/X7", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y7", 'rb') as file:
    y = pickle.load(file)

X = truncate_to_shortest_and_convert_to_array(X)
print(y)
y = normalize_y(y)
print(y)
# Seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.05, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.05, random_state=seed)
with open('dataCNN/X8','wb') as file:
    pickle.dump(X_test,file)

with open('dataCNN/y8','wb') as file:
    pickle.dump(y_test, file)
print(y_test.shape)
batch_size = 100
train_dataloader = MyDataLoader(X_train, y_train, batch_size)
val_dataloader = MyDataLoader(X_val, y_val, batch_size)
test_dataloader = MyDataLoader(X_test, y_test, batch_size)
