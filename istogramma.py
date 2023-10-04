import matplotlib.pyplot as plt
import pickle

with open("./dataCNN/y2", 'rb') as file:
    y = pickle.load(file)
p = y[:,0]
q = y[:,1]
r = y[:,2]
alfa = y[:,3]
beta = y[:,4]
gamma = y[:,5]
plt.hist(p)
figura, grafico = plt.subplots(2,3, sharex=True)
grafico[0,0].hist(p, density=True)
grafico[0,0].set_xlabel("$rad/s$")
grafico[0,0].set_ylabel("Frequenza")
grafico[0,0].set_title("Angular velocity p")
grafico[0,0].xaxis.labelpad = 10
grafico[0,1].hist(q, density=True)
grafico[0,1].set_xlabel("$rad/s$")
grafico[0,1].set_title("Angular velocity q")
grafico[0,1].xaxis.labelpad = 10
grafico[0,2].hist(r, density=True)
grafico[0,2].set_xlabel("$rad/s$")
grafico[0,2].set_title("Angular velocity r")
grafico[0,2].xaxis.labelpad = 10
grafico[1,0].hist(alfa, density=True)
grafico[1,0].set_xlabel("$rad$")
grafico[1,0].set_ylabel("Frequenza")
grafico[1,0].set_title("Euler angles $\\alpha$")
grafico[1,1].hist(beta, density=True)
grafico[1,1].set_xlabel("$rad$")
grafico[1,1].set_title("Euler angles $\\beta$")
grafico[1,2].hist(gamma, density=True)
grafico[1,2].set_xlabel("$rad$")
grafico[1,2].set_title("Euler angles $\\gamma$")
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.tight_layout()
plt.show()