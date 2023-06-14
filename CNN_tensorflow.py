import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten

# Imposta il seme per la generazione dei numeri casuali
seed = 42
np.random.seed(seed)

# Carica il database da file
X = np.load('X.npy')
y = np.load('y.npy')

# Dividi il dataset in addestramento e verifica in modo casuale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

# Crea il modello della rete neurale
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(2400, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(6))  # Output layer

# Compila il modello
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape dei dati di input per adattarli al formato richiesto dalla convoluzione 1D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Addestra il modello
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Valuta il modello sui dati di verifica
loss = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
