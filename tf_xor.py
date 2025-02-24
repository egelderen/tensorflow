import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Build a slightly larger model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=8, activation='relu'),  # More neurons
    tf.keras.layers.Dense(units=4, activation='relu'),  # Extra layer
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train longer
model.fit(X, y, epochs=200, batch_size=1, verbose=0)

# Test predictions
predictions = model.predict(X, verbose=0)
print("Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions[i][0]:.4f} (Actual: {y[i][0]})")

# Model summary
model.summary()