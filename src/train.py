import numpy as np
from keras import models, layers
from keras.datasets import boston_housing

# Load dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalize the data
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Build the model
def build_model():
    """
    Builds a Keras regression model for predicting housing prices.
    """
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # Single output for regression
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Hyperparameters
k = 4  # Number of folds
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

# K-Fold Cross-Validation
for i in range(k):
    print(f"Processing fold #{i + 1}")
    
    # Prepare validation data
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    # Prepare training data
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )
    
    # Build and train the model
    model = build_model()
    history = model.fit(
        partial_train_data,
        partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=1,
        verbose=0
    )
    
    # Record validation MAE history
    mae_history = history.history['val_mae']  # Note: Metric name may vary based on Keras version
    all_mae_histories.append(mae_history)

# Calculate the average MAE history across all folds
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

# Save the results for analysis
np.save("results/average_mae_history.npy", average_mae_history)

# Plot the average MAE history
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.title('Validation MAE vs Epochs')
plt.show()

# Smooth the curve for better visualization
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])  # Ignore the first 10 points

# Plot the smoothed curve
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.title('Smoothed Validation MAE vs Epochs')
plt.show()
