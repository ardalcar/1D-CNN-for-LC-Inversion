import os
import numpy as np
import pickle
import sys
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

def load_LC(data_folder, num_samples):
    
    X = []
    y = []

    for i in range(num_samples):
        filename = f"CdL_Sentinel{i}.txt"
        file_path = os.path.join(current_path, data_folder, filename)

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                lines = [np.float32(line.strip()) for line in lines]
                input_dim = len(lines)

                output = lines[:output_dim]  # First 6 values are the output
                input_data = lines[output_dim:input_dim]  # Next values are input

                X.append(input_data)
                y.append(output)
        except FileNotFoundError:
            # print(f"File not found::{file_path}. Skipping...")
            continue
    
    return X, y

def truncate_to_shortest_and_convert_to_array(light_curves):
    # Trova la lunghezza della curva più corta
    min_length = min(len(curve) for curve in light_curves)

    # Tronca tutte le curve alla lunghezza della curva più corta e le converte in un array
    truncated_curves = [curve[:min_length] for curve in light_curves]
    array_curves = np.array(truncated_curves, dtype = np.float64)

    return array_curves

def normalize_y(y):
    y = np.array(y, dtype=np.float64)
    y_vel = y[:,:3]
    y_ang = y[:,-3:]
    y_v_n = y_vel/0.0002
    y_a_n = y_ang/(np.pi)
    y_norm = np.column_stack([y_v_n,y_a_n])
    return y_norm

def generate_dataset(data_folder, num_samples):
    X, y = load_LC(data_folder, num_samples)
    X = truncate_to_shortest_and_convert_to_array(X)
    y = np.array(y, dtype=np.float64)

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    return X, y


def grid_selection(X, y, num_points=200):
    # Prendo solo gli ultimi 3 valori di y per la griglia
    y_grid = y[:, -3:]
    # Numero di punti per asse 
    num_points_per_axis = int(np.cbrt(num_points)) + 1

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
    X = selected_X
    y = selected_y
    print(f'Numero di punti selezionati: {len(selected_X)}')
    X = np.array(X, dtype=np.float64)
    y = normalize_y(y)
    return X, y

def save_dataset(X, y, X9_path, y9_path):

    with open(X9_path,'wb') as file:
        pickle.dump(X, file)

    with open(y9_path,'wb') as file:
        pickle.dump(y, file)
    print(y.shape)
    print(X.shape)

#def euler_to_quaternion(y_angle):
#    rotation = R.from_euler('xyz', y_angle)
#    quaternion = rotation.as_quat()
#    return quaternion
#
#def normalize_y_vel(y_vel, max_vel=0.0002):
#    y_norm_vel=y_vel/max_vel
#    return y_norm_vel
#
#def transform_y(y):
#    y_angle = y[:,-3:]
#    y_vel = y[:,:3]
#    y_angle_quat = euler_to_quaternion(y_angle)
#    y_vel_norm = normalize_y_vel(y_vel)
#    y_norm = np.column_stack([y_vel_norm, y_angle_quat])
#    y_norm = np.array(y_norm, dtype = np.float64)
#    return y_norm

########################################## main ######################################

current_path = os.getcwd()
data_folder = 'X7_dataset'

if len(sys.argv) > 1:
    data_folder = sys.argv[1]

output_dim = 6
num_samples = 10000
num_grid_points = 400

X, y = generate_dataset(data_folder, num_samples)

X, y = grid_selection(X, y, num_points = num_grid_points)

# Risultati
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

np.set_printoptions(linewidth=np.inf)
print('shape and value of y:')
print(y)

X9p='X9'
y9p='y9'

dataset_path = 'new_dataset'

if len(sys.argv) > 2:
    data_folder = sys.argv[2]

X9_path=os.path.join(current_path, dataset_path, X9p)
y9_path=os.path.join(current_path, dataset_path, y9p)

os.makedirs(os.path.join(current_path, data_folder), exist_ok=True)

save_dataset(X, y, X9_path, y9_path)
