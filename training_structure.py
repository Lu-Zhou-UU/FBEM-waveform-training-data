import numpy as np
import tensorflow as tf
from scipy.interpolate import PPoly
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Load data
# Assuming a Python equivalent for MATLAB's .mat file loading
from scipy.io import loadmat

data = loadmat('FBEM-master/FBEM_trainingdata.mat')
sigma_0_snow_surf = data['sigma_0_snow_surf']
sigma_0_snow_vol = data['sigma_0_snow_vol']
sigma_0_ice_surf = data['sigma_0_ice_surf']
h_s = data['h_s']
P_t_ml = data['P_t_ml']

# Initialize variables
theta = np.logspace(np.log10(1e-6), np.log10(np.pi / 2), 200)
Y1, Y2, Y3, Y4, idxx = [], [], [], [], []

# Main loop
for i in range(len(sigma_0_ice_surf)):
    # Interpolation and averaging
    test = PPoly(sigma_0_snow_surf[i]).__call__(theta)
    Y1.append(np.mean(np.real(test[90:180])))
    test = PPoly(sigma_0_snow_vol[i]).__call__(theta)
    Y2.append(np.mean(np.real(test[90:180])))
    Y3.append(h_s[i])
    test = PPoly(sigma_0_ice_surf[i]).__call__(theta)
    Y4.append(np.mean(np.real(test[90:180])))
    
    test = P_t_ml[:, i]
    if np.any(test < 0):
        idxx.append(i)
    else:
        # Assuming X is initialized as a 4D array
        X[0, :, 0, i] = test

# Remove invalid indices
Y1, Y2, Y3, Y4 = np.array(Y1), np.array(Y2), np.array(Y3), np.array(Y4)
invalid_idx = np.where(Y3 <= 0)[0]
invalid_idx = np.unique(np.concatenate((invalid_idx, idxx)))

# Ensure indices are within bounds and remove invalid entries
Y1 = np.delete(Y1, invalid_idx)
Y2 = np.delete(Y2, invalid_idx)
Y3 = np.delete(Y3, invalid_idx)
Y4 = np.delete(Y4, invalid_idx)
X = np.delete(X, invalid_idx, axis=3)

# Normalize the input data
X = tf.keras.utils.normalize(X, axis=1)

# Concatenate Y into a single output array
Y = np.column_stack((Y1, Y2, Y3, Y4))

# Define the neural network in TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1, 70, 1)),
    tf.keras.layers.Conv2D(64, (1, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
    tf.keras.layers.Conv2D(128, (1, 6), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
    tf.keras.layers.Conv2D(256, (1, 6), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X[:, :, :, :6000], Y[:6000],
    validation_data=(X[:, :, :, 6000:], Y[6000:]),
    epochs=50,
    batch_size=100
)

# Generate predictions
predictions = model.predict(X)

# Rescale predictions to physical ranges (if needed)
predictions = tf.keras.utils.normalize(predictions, axis=1)

# Plot results
plt.figure(figsize=(12, 8))
for i, (label, color) in enumerate(zip(
    ['Snow surface scattering', 'Snow volume scattering', 'Ice surface scattering', 'Snow depth'],
    ['purple', 'red', 'green', 'blue']
)):
    plt.subplot(2, 2, i+1)
    density, bins = np.histogram(predictions[:, i], bins=100, density=True)
    plt.plot(bins[:-1], density, '-', linewidth=2, color=color)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel(label)
plt.tight_layout()
plt.show()
