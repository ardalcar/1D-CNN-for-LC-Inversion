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

def quaternion_to_euler(w, x, y, z):
    """
    Convert a quaternion into Euler angles (yaw, pitch, and roll)

    Parameters:
    w, x, y, z (float): Components of the quaternion

    Returns:
    tuple: Euler angles in the format (yaw, pitch, roll)
    """
    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # Use 90 degrees if out of range
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    return yaw, pitch, roll

# Example usage
w, x, y, z = 0.723, 0.532, 0.392, 0.201  # Example quaternion components
yaw, pitch, roll = quaternion_to_euler(w, x, y, z)


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


