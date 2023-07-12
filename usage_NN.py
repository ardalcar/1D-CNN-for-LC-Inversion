import CNN_Py2 as P2
import torch
import matplotlib.pyplot as plt


modello_addestrato = P2.NeuralNetwork()
modello_addestrato.load_state_dict(torch.load("modello_addestrato.pth"))

input_da_valutare = torch.from_numpy(P2.X_train).unsqueeze(1).float()
input_da_valutare = input_da_valutare.to(P2.device)

modello_addestrato.eval()
previsioni = modello_addestrato(input_da_valutare)

y_pred = previsioni

# Creazione dei grafici
for i in range(6):
    plt.figure()
    plt.plot(P2.y_true[:, i], label='y_true')
    plt.plot(y_pred[:, i], label='y_pred')
    plt.xlabel('Sample')
    plt.ylabel(f'Output {i+1}')
    plt.legend()
    nome_file=f"grafico_{i}.png"
    plt.savefig(nome_file)
    plt.clf()

# Mostra i grafici
del input_da_valutare
del previsioni
plt.show()