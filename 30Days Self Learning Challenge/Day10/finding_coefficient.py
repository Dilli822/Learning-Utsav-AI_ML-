import numpy as np

# Dataset
Hours = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
Pass = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
             1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# Parameters
alpha = 0.1  # Learning rate
threshold = 1e-6  # Convergence threshold
iterations = 10000  # Maximum number of iterations

# Initialize coefficients
beta_0 = 0.0
beta_1 = 0.0

for i in range(iterations):
    # Compute predicted probabilities
    linear_combination = beta_0 + beta_1 * Hours  # z = β0 + β1 * Hours
    p = 1 / (1 + np.exp(-linear_combination))  # p = 1 / (1 + e^(-z))

    # Print the logistic function and predicted probabilities
    print(f"\nIteration {i + 1}:")
    print(f"Linear Combination (z): {linear_combination}")
    print(f"Predicted Probabilities (p): {p}")

    # Calculate gradients
    gradient_0 = np.sum(p - Pass) / len(Pass)  # Gradient for β0
    gradient_1 = np.sum((p - Pass) * Hours) / len(Pass)  # Gradient for β1

    # Print gradients
    print(f"Gradient for Beta_0: {gradient_0}")
    print(f"Gradient for Beta_1: {gradient_1}")

    # Update coefficients
    # Update formula: β0 = β0 - α * (average gradient for β0)
    beta_0_new = beta_0 - alpha * gradient_0
    # Update formula: β1 = β1 - α * (average gradient for β1)
    beta_1_new = beta_1 - alpha * gradient_1

    # Print updates
    print(f"Update for Beta_0: {beta_0_new:.6f} = {beta_0:.6f} - {alpha} * {gradient_0:.6f}")
    print(f"Update for Beta_1: {beta_1_new:.6f} = {beta_1:.6f} - {alpha} * {gradient_1:.6f}")

    # Check for convergence
    if abs(beta_0_new - beta_0) < threshold and abs(beta_1_new - beta_1) < threshold:
        break

    beta_0, beta_1 = beta_0_new, beta_1_new

print(f"\nConverged after {i + 1} iterations")
print(f"Beta_0 (Intercept): {beta_0:.6f}")
print(f"Beta_1 (Coefficient for Hours): {beta_1:.6f}")
