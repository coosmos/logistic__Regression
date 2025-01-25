import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Standardize Data
X = np.loadtxt('logisticX.csv', delimiter=',')
y = np.loadtxt('logisticY.csv', delimiter=',')

# Standardize the data (mean = 0, std = 1)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_standardized = (X - mean) / std

# Step 2: Define Logistic Regression Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -(1/m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
        cost_history.append(cost_function(X, y, theta))
    
    return theta, cost_history

# Step 3: Train Logistic Regression Model
# Add a column of ones for the bias term
X_with_bias = np.hstack((np.ones((X_standardized.shape[0], 1)), X_standardized))
theta = np.zeros(X_with_bias.shape[1])  # Initialize theta

# Train the model with a learning rate of 0.1
learning_rate = 0.1
iterations = 1000
theta, cost_history = gradient_descent(X_with_bias, y, theta, learning_rate, iterations)

print("Theta after convergence:", theta)
print("Final cost function value:", cost_history[-1])

# Step 4: Plot Cost Function vs Iteration
plt.plot(range(len(cost_history)), cost_history, label='Learning Rate: 0.1')
plt.title('Cost Function vs Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.legend()
plt.show()

# Step 5: Plot Dataset with Decision Boundary

# Decision Boundary
x1 = np.linspace(min(X_standardized[:, 0]), max(X_standardized[:, 0]), 100)
x2 = -(theta[0] + theta[1]*x1) / theta[2]

# Plot the data as line plots with decision boundary
for i in np.unique(y):
    plt.plot(X_standardized[y == i, 0], X_standardized[y == i, 1], label=f'Class {int(i)}', marker='o')

plt.plot(x1, x2, label='Decision Boundary', color='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Dataset with Decision Boundary')
plt.show()

# Step 6: Train Models with Different Learning Rates
learning_rates = [0.1, 5]
iterations = 100

for lr in learning_rates:
    theta = np.zeros(X_with_bias.shape[1])
    _, cost_history = gradient_descent(X_with_bias, y, theta, lr, iterations)
    plt.plot(range(len(cost_history)), cost_history, label=f'Learning Rate: {lr}')

plt.title('Cost Function vs Iteration for Different Learning Rates')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.legend()
plt.show()

# Step 7: Confusion Matrix and Metrics
# Predictions
predictions = sigmoid(np.dot(X_with_bias, theta)) >= 0.5

# Confusion Matrix
tp = np.sum((predictions == 1) & (y == 1))  # True Positive
tn = np.sum((predictions == 0) & (y == 0))  # True Negative
fp = np.sum((predictions == 1) & (y == 0))  # False Positive
fn = np.sum((predictions == 0) & (y == 1))  # False Negative

confusion_matrix = np.array([[tp, fp], [fn, tn]])
print("Confusion Matrix:\n", confusion_matrix)

# Metrics
accuracy = (tp + tn) / len(y)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
