import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate some data points for training and testing sets
np.random.seed(0)
train_data = np.random.randn(100, 2) + np.array([0, 0])
test_data = np.random.randn(100, 2) + np.array([4, 4])

# Create a figure for better organization
plt.figure(figsize=(8, 6))

# Fit curves for training and testing sets using kdeplot
sns.kdeplot(x=train_data[:, 0], y=train_data[:, 1], color='blue', label='Training Data - Curve', fill=True, alpha=0.3)
sns.kdeplot(x=test_data[:, 0], y=test_data[:, 1], color='red', label='Testing Data - Curve', fill=True, alpha=0.3)

# Plot the datasets as scatter plots
plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', label='Training Data', s=20)
plt.scatter(test_data[:, 0], test_data[:, 1], color='red', label='Testing Data', s=20)

# Add title and legend
plt.title('Training vs Testing Data with Curves')
plt.legend()
plt.show()

# Simulating data for loss and cost functions
iterations = np.arange(1, 101)
loss = np.log(iterations + 1)
cost = np.sqrt(iterations)  # Example of a cost function that decreases

# Plotting both functions
plt.figure(figsize=(8, 6))

# Plotting the loss function
plt.plot(iterations, loss, label="Loss Function", color='purple')

# Plotting the cost function
plt.plot(iterations, cost, label="Cost Function", color='green')

# Adding labels, title, and grid
plt.title('Loss and Cost Functions over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.grid(True)
plt.legend()

# Display the plot
plt.show()

from mpl_toolkits.mplot3d import Axes3D

def cost_function(x, y):
    return x**2 + y**2

# Generate mesh grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = cost_function(X, Y)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Gradient Descent Visualization')
plt.show()


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression

# Logistic Regression
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1)
model = LogisticRegression().fit(X, y)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue')
x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(x_vals, model.predict_proba(x_vals)[:, 1], color='red', label='Logistic Regression')
plt.title('Logistic Regression')
plt.legend()
plt.show()

# Linear Regression
X = np.random.rand(100, 1)
y = 2 * X + np.random.randn(100, 1) * 0.1
model = LinearRegression().fit(X, y)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='green', label='Linear Regression')
plt.title('Linear Regression')
plt.legend()
plt.show()

