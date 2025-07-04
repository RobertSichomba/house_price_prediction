import numpy as np
from keras.datasets import boston_housing
from keras import models, layers
import matplotlib.pyplot as plt

# Load the dataset
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

# Perform k-fold cross-validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

# Calculate average MAE history
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

# Smooth the MAE curve
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

# Plot the validation MAE
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.title('Validation MAE Over Epochs')
plt.show()

# Train the final model
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)

# Evaluate the model on test data
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

# Display the test results
print(f"Test MSE: {test_mse_score}")
print(f"Test MAE: {test_mae_score}")

# Visualize predictions vs actual targets
predictions = model.predict(test_data).flatten()
plt.scatter(test_targets, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.plot([0, 50], [0, 50], 'k--', color='red')  # Line y=x for reference
plt.show()

# Save the model
model.save('my_model.keras')

