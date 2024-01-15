import matplotlib.pyplot as plt
import pickle

with open("./dataCNN/y3", 'rb') as file:
    y = pickle.load(file)
p = y[:,0]
q = y[:,1]
r = y[:,2]
alfa = y[:,3]
beta = y[:,4]
gamma = y[:,5]

plt.figure(1)
plt.hist(p, density=True)
plt.xlabel("$rad/sq$")
plt.ylabel("Frequenza")
plt.title("Angular velocity p")

plt.figure(2)
plt.hist(q, density=True)
plt.xlabel("$rad/sq$")
plt.title("Angular velocity q")

plt.figure(3)
plt.hist(r, density=True)
plt.xlabel("$rad/sq$")
plt.title("Angular velocity r")

plt.figure(4)
plt.hist(alfa, density=True)
plt.xlabel("$rad$")
plt.ylabel("Frequenza")
plt.title("Euler angles $\\alpha$")

plt.figure(5)
plt.hist(beta, density=True)
plt.xlabel("$rad$")
plt.ylabel("Frequenza")
plt.title("Euler angles $\\beta$")

plt.figure(6)
plt.hist(gamma, density=True)
plt.xlabel("$rad$")
plt.ylabel("Frequenza")
plt.title("Euler angles $\\gamma$")

plt.show()