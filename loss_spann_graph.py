import matplotlib.pyplot as plt
import numpy as np

def Loss_graph(test, val, title):

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
    plt.plot(data_mean,label='Loss Train moving average', color='#A2231D', linestyle='--', linewidth=2)
    plt.plot(data_test, label='Validation set', color='blue')
    plt.plot(data_test_mean,label='Loss Validation moving average', color='#0ABAB5', linestyle='--', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    save_path=title.replace(" ", "_")
    plt.savefig(save_path)


test = 'loss_spannRNN5.txt'  
val = 'loss_spannRNN5_test.txt'
title = 'RNN Loss graph'
Loss_graph(test, val, title)

plt.clf()

test = 'loss_spannGRU.txt'  
val = 'loss_spannGRU_val.txt'

title = 'GRU Loss graph'
Loss_graph(test, val, title)