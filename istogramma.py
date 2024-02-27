import matplotlib.pyplot as plt
import pickle

with open("./new_dataset/y9", 'rb') as file:
    y = pickle.load(file)

# Definizione delle variabili
p = y[:,0]
q = y[:,1]
r = y[:,2]
alfa = y[:,3]
beta = y[:,4]
gamma = y[:,5]

# Creazione della figura e degli assi (subplots)
fig, axs = plt.subplots(2, 3, figsize=(15, 10)) # figsize è opzionale, regola le dimensioni della figura

# Primo subplot
axs[0, 0].hist(p, density=True)
axs[0, 0].set_xlabel("$rad/s$")
axs[0, 0].set_ylabel("Frequenza")
axs[0, 0].set_title("Angular velocity p")

# Secondo subplot
axs[0, 1].hist(q, density=True)
axs[0, 1].set_xlabel("$rad/s$")
axs[0, 1].set_title("Angular velocity q")

# Terzo subplot
axs[0, 2].hist(r, density=True)
axs[0, 2].set_xlabel("$rad/s$")
axs[0, 2].set_title("Angular velocity r")

# Quarto subplot
axs[1, 0].hist(alfa, density=True)
axs[1, 0].set_xlabel("$rad$")
axs[1, 0].set_ylabel("Frequenza")
axs[1, 0].set_title("Euler angles $\\alpha$")

# Quinto subplot
axs[1, 1].hist(beta, density=True)
axs[1, 1].set_xlabel("$rad$")
axs[1, 1].set_ylabel("Frequenza")
axs[1, 1].set_title("Euler angles $\\beta$")

# Sesto subplot
axs[1, 2].hist(gamma, density=True)
axs[1, 2].set_xlabel("$rad$")
axs[1, 2].set_ylabel("Frequenza")
axs[1, 2].set_title("Euler angles $\\gamma$")

plt.tight_layout() # Questo migliora l'aspetto complessivo, prevenendo sovrapposizioni
plt.show()
