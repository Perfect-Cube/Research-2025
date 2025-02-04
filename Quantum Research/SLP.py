import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 2) * np.array([100, 10])
threshold_speed = 50
y = np.where(X[:, 0] > threshold_speed, 1, 0)

# Normalize features
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# Initialize weights and bias
np.random.seed(1)
weights = np.random.rand(2)
bias = np.random.rand(1)

# Activation function
def activation(x):
    return np.where(x >= 0, 1, 0)

# Training parameters
learning_rate = 0.1
epochs = 100

# Training loop
for epoch in range(epochs):
    output = activation(np.dot(X_normalized, weights) + bias)
    error = y - output
    weights += learning_rate * np.dot(X_normalized.T, error)
    bias += learning_rate * np.sum(error, keepdims=True)
    print(f'Epoch {epoch+1}, Error: {np.mean(np.abs(error))}')

# Test the model
predicted = activation(np.dot(X_normalized, weights) + bias)
accuracy = np.mean(predicted == y)
print(f'Training Accuracy: {accuracy:.2f}')

# Visualize the decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y, cmap='bwr', alpha=0.5)
plt.xlabel('Normalized Speed')
plt.ylabel('Normalized Acceleration')
plt.title('Vehicle Movement Classification')
plt.show()
