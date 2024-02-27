import pickle
import numpy as np
import os
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


# Funzione per caricare i dati da un file pickle
def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

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

# Carica i dati

current_path = os.getcwd()
fc='FullyConnected2'
dataset_path='dataCNN'
X7p='X7'
y7p='y7'
X7_path=os.path.join(current_path, fc, dataset_path, X7p)
y7_path=os.path.join(current_path, fc, dataset_path, y7p)

X = load_data(X7_path)
y = load_data(y7_path)

# Prendo solo gli ultimi 3 valori di y per la griglia
y_grid = y[:, -3:]

# Numero di punti per asse 
num_points_per_axis = int(np.cbrt(200)) + 1

# Creazione griglia 3D
x_range = np.linspace(-np.pi/3, np.pi/3, num_points_per_axis)
y_range = np.linspace(-np.pi/6, np.pi/6, num_points_per_axis)
z_range = np.linspace(-np.pi/3, np.pi/3, num_points_per_axis)
grid = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

# Utilizzo di cKDTree per trovare i punti più vicini
tree = cKDTree(y_grid)
_, indices = tree.query(grid, k=1)

# Rimozione duplicati 
unique_indices = np.unique(indices)

# Estrazione dei dati
selected_X = [X[i] for i in unique_indices]
selected_y = y[unique_indices]
print(selected_X.__len__)
print(selected_y.shape)
print(selected_y)

# Risultati
print(f'Numero di punti selezionati: {len(selected_X)}')

X = truncate_to_shortest_and_convert_to_array(selected_X)
print(X.shape)

y = transform_y(selected_y)
np.set_printoptions(linewidth=np.inf)
print('shape and value of y:')
print(y)

X9p='X9'
y9p='y9'
X9_path=os.path.join(current_path, fc, dataset_path, X9p)
y9_path=os.path.join(current_path, fc, dataset_path, y9p)

with open(X9_path,'wb') as file:
    pickle.dump(X,file)

with open(y9_path,'wb') as file:
    pickle.dump(y, file)
print(y.shape)
print(X.shape)
