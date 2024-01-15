import matplotlib.pyplot as plt

loss_spann = []
with open('loss_spann.txt') as file:
    for riga in file:
        loss = float(riga.strip())
        loss_spann.append(loss)

loss_spann2 = loss_spann[80:]
loss_spann3 = loss_spann[420:]
range1=range(len(loss_spann))
range2=range1[80:]
range3=range1[420:]

plt.figure()

plt.subplot(3,1,1)
plt.title('Valore della Loss nelle varie epoche')
plt.plot(range1,loss_spann)

plt.subplot(3,1,2)
plt.plot(range2,loss_spann2)

plt.subplot(3,1,3)
plt.plot(range3,loss_spann3)
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.tight_layout()

plt.savefig('loss_graph.png')
plt.show()