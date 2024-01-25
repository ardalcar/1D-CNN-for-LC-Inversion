import math
import pickle
import numpy as np

def euler_to_quaternion(yaw, pitch, roll):
    """
    Convert Euler angles to quaternion.

    Parameters:
    yaw (float): Rotation angle around the z-axis (in radians)
    pitch (float): Rotation angle around the y-axis (in radians)
    roll (float): Rotation angle around the x-axis (in radians)

    Returns:
    tuple: Quaternion in the format (w, x, y, z)
    """
    # Calculate cosines and sines of half angles
    c1 = math.cos(yaw / 2)
    c2 = math.cos(pitch / 2)
    c3 = math.cos(roll / 2)
    s1 = math.sin(yaw / 2)
    s2 = math.sin(pitch / 2)
    s3 = math.sin(roll / 2)

    # Calculate components of the quaternion
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * s2 * c3 + c1 * c2 * s3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3

    return (w, x, y, z)

with open("dataCNN/y41", 'rb') as file:
    y=pickle.load(file)

y42=[]

for row in y:
    # Seleziona gli ultimi tre elementi della riga e aggiungili alla lista
    a, b, c = row[-3:]
    d = euler_to_quaternion(a, b, c)
    y42.append(d)

# Converti la lista in un array NumPy se necessario
selected_elements_array = np.array(y42)

with open("dataCNN/y42", 'wb') as file:
    pickle.dump(selected_elements_array, file, pickle.HIGHEST_PROTOCOL)


