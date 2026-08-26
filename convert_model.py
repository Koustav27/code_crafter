import tensorflow as tf
import h5py
import numpy as np

print("Building model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),

    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 3
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 4
    tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 5
    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1500, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(38, activation="softmax")
])

print("Model built successfully.")
print("Loading weights manually...")

# Open the Keras 3 weights file
with h5py.File("model.weights.h5", "r") as f:

    # Get Conv2D layers
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    # Load 10 convolution layers
    for i, layer in enumerate(conv_layers):

        weights = f[f"layers/conv2d{'_' + str(i) if i > 0 else ''}/vars/0"][:]
        bias = f[f"layers/conv2d{'_' + str(i) if i > 0 else ''}/vars/1"][:]

        layer.set_weights([weights, bias])

        print(f"Loaded Conv2D layer {i + 1}/10")

    # Dense layers
    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]

    # Dense 1500
    weights = f["layers/dense/vars/0"][:]
    bias = f["layers/dense/vars/1"][:]

    dense_layers[0].set_weights([weights, bias])

    print("Loaded Dense layer 1/2")

    # Dense 38
    weights = f["layers/dense_1/vars/0"][:]
    bias = f["layers/dense_1/vars/1"][:]

    dense_layers[1].set_weights([weights, bias])

    print("Loaded Dense layer 2/2")

print()
print("ALL WEIGHTS LOADED SUCCESSFULLY!")

print("Saving model...")

model.save("plant_disease_model_fixed.h5")

print()
print("======================================")
print("MODEL CONVERSION SUCCESSFUL!")
print("Saved as: plant_disease_model_fixed.h5")
print("======================================")