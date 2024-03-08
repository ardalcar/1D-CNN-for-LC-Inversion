import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def truncate_to_shortest_and_convert_to_array(light_curves):
    # Trova la lunghezza della curva più corta
    min_length = min(len(curve) for curve in light_curves)

    # Tronca tutte le curve alla lunghezza della curva più corta e le converte in un array
    truncated_curves = [curve[:min_length] for curve in light_curves]
    array_curves = np.array(truncated_curves)

    return array_curves

def euler_to_quaternion(y_angle):
    rotation = R.from_euler('xyz', y_angle)
    quaternion = rotation.as_quat()
    return quaternion

def normalize_y_vel(y_vel, max_vel=0.0002):
    y_norm_vel=y_vel/max_vel
    return y_norm_vel

def transform_y(y):
    y_angle = y[:,-3:]
    y_vel = y[:,:3]
    y_angle_quat = euler_to_quaternion(y_angle)
    y_vel_norm = normalize_y_vel(y_vel)
    y_norm = np.column_stack([y_vel_norm,y_angle_quat])
    y_norm = np.array(y_norm, dtype=float)
    return y_norm

################################### main ###################################

with open("./dataCNN/X7", 'rb') as file:
    X = pickle.load(file)

with open("./dataCNN/y7", 'rb') as file:
    y = pickle.load(file)

X = truncate_to_shortest_and_convert_to_array(X)
print(X.shape)

y = transform_y(y)
np.set_printoptions(linewidth=np.inf)
print('shape and value of y:')
print(y)

with open('X8','wb') as file:
    pickle.dump(X,file)

with open('y8','wb') as file:
    pickle.dump(y, file)
print(y.shape)

