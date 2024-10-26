import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Feature values
y = np.sin(X) + np.random.normal(0, 0.5, X.shape)  # Target values with noise

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define λ values for the plots
lambdas = [0, 10, 1000]  # No regularization, small λ, large λ
degrees = 15

# Create plots for each λ
x_range = np.linspace(0, 10, 100).reshape(-1, 1)  # For plotting smooth lines
for lambda_val in lambdas:
    # Polynomial features
    poly = PolynomialFeatures(degree=degrees)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    x_poly_range = poly.transform(x_range)  # Transform x_range for predictions

    # Ridge regression model
    model = Ridge(alpha=lambda_val)
    model.fit(X_poly_train, y_train)

    # Predicting on the range for smooth line
    y_pred_range = model.predict(x_poly_range)

    # Plotting results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='white', label='Training Data', alpha=0.5)  # Training data as points
    plt.scatter(X_test, y_test, color='red', label='Test Data', alpha=0.5)  # Test data as points
    plt.plot(x_range, y_pred_range, color='black', label='Model Prediction (λ={})'.format(lambda_val))  # Predictions as a line
    plt.title('Regularization with λ={}'.format(lambda_val))
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()
