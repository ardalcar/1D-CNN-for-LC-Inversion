import os
import numpy as np
import pickle

current_path = os.getcwd()
fc='FullyConnected2'
data_folder = 'X7_dataset'
output_dim = 6
num_samples = 10000

X = []
y = []

for i in range(num_samples):
    filename = f"CdL_Sentinel{i}.txt"
    file_path = os.path.join(current_path, fc, data_folder, filename)
    
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

input_dim = len(X[0])

# Convert X and y to numpy arrays of dtype np.float32
X = np.array(X, dtype = object)
y = np.array(y, dtype = np.float32)

# Print the shapes of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

dataset_path='dataCNN'
X7p='X7'
y7p='y7'
X7_path=os.path.join(current_path, fc, dataset_path, X7p)
y7_path=os.path.join(current_path, fc, dataset_path, y7p)

with open(X7_path, 'wb') as file:
    pickle.dump(X, file)

with open(y7_path, 'wb') as file:
    pickle.dump(y, file)


