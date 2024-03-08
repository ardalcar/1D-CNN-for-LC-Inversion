import os
import numpy as np
import pickle
import sys
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
import torch

def load_LC(data_folder, num_samples):
    
    X = []
    y = []

    for i in range(num_samples):
        filename = f"CdL_Sentinel{i}.txt"
        file_path = os.path.join(data_folder, filename)
        output_dim=6

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

def crea_sottodataset_stratificati(data, y_labels, dimensioni_sottodataset):
    if not isinstance(dimensioni_sottodataset, list):
        dimensioni_sottodataset = [dimensioni_sottodataset]
    sottodatasetX = {}
    sottodatasety = {}
    
    # Per ogni dimensione richiesta, si creano sottodataset stratificati
    for dimensione in dimensioni_sottodataset:

        # Stratificazione dei dati basata su y_labels per gli angoli di Eulero
        
        if dimensione <= 27:
            num_bins = 2
        elif dimensione <= 64:
            num_bins = 3
        elif dimensione <= 125:
            num_bins = 4
        elif dimensione <= 216:
            num_bins = 5
        elif dimensione < 343:
            num_bins = 6
        elif dimensione < 512:
            num_bins = 7
        else:
            num_bins = 8
         
        bin_data = pd.DataFrame()
        for i, angolo in enumerate(['alfa', 'beta', 'gamma']):
            # Creazione dei bin per ogni angolo di Eulero
            bin_data['bin_' + angolo] = pd.qcut(y_labels[:, i + 3], q=num_bins, labels=False, duplicates='drop')

        if any(bin_data['bin_' + angolo].value_counts().min() < 2 for angolo in ['alfa', 'beta', 'gamma']):
            raise ValueError("Non ci sono abbastanza campioni per stratificare i dati con i bin attuali.")

        # Campionamento stratificato per ogni dimensione
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        _, X_campione, _, y_campione = train_test_split(data, y_labels, test_size=dimensione, stratify=bin_data, random_state=seed)
        sottodatasetX[dimensione] = X_campione
        sottodatasety[dimensione] = y_campione

    return sottodatasetX, sottodatasety

if __name__ == "__main__":
    if len(sys.argv) > 3:
        data_folder = sys.argv[1]
        # Converte la stringa del parametro in un intero o una lista di interi
        dimensioni_input = sys.argv[3]
        if ',' in dimensioni_input:
            dimensioni_sottodataset = list(map(int, dimensioni_input.split(',')))
        else:
            dimensioni_sottodataset = int(dimensioni_input)

    num_samples = 10000

    X, y = generate_dataset(data_folder, num_samples)

    sottodataset_X, sottodataset_y = crea_sottodataset_stratificati(X, y, dimensioni_sottodataset)

    destination_path = sys.argv[2]
    x_path = 'X'
    y_path = 'y'


    for i in sottodataset_y:
        dd=str(i)
        Xi = sottodataset_X[i]
        yi = sottodataset_y[i]    

        xx=x_path+dd
        yy=y_path+dd
        pathX = os.path.join('.', destination_path, xx)
        pathy = os.path.join('.', destination_path, yy)
        with open(pathX,'wb') as file:
            pickle.dump(Xi, file)

        with open(pathy,'wb') as file:
            pickle.dump(yi, file)