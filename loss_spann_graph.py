import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft

test = 'loss_spannRNN5.txt'  
val = 'loss_spannRNN5_test.txt'

file_path = test
with open(file_path, 'r') as file:
    data = [float(line.strip()) for line in file]

file_path2 = val 
with open(file_path2, 'r') as file:
    data_test = [float(line.strip()) for line in file]


window_size=50

data_mean=np.convolve(data, np.ones(window_size)/window_size, mode='valid')
data_test_mean=np.convolve(data_test, np.ones(window_size)/window_size, mode='valid')

# Crea un grafico
plt.plot(data, label='Train set', color='red', linewidth=1)
plt.plot(data_mean,label='Loss Train moving average', color='#FFB7C5', linestyle='--', linewidth=2)
plt.plot(data_test, label='Test set', color='blue')
plt.plot(data_test_mean,label='Loss Test moving average', color='#0ABAB5', linestyle='--', linewidth=2)
plt.title('RNN Loss graph')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('grafico_loss_RNN')
plt.show()

