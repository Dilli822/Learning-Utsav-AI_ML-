import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

data = pd.read_csv(url, names=columns, na_values="?")

# Display the first few rows of the dataset
print(data.head())

# Handle missing values (drop rows with NaN values for simplicity)
data = data.dropna()

# The target variable 'target' has values 0 (no heart disease) or 1-4 (levels of heart disease). We'll simplify it to binary classification:
data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)

# Separate features (X) and target (y)
X = data.drop("target", axis=1)
y = data["target"]

# Step 2: Preprocess the Data
# Standardize the feature values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train the Model Without Splitting the Data

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the entire dataset without splitting
model.fit(X_scaled, y)

# Predict on the same dataset (no test set)
y_pred = model.predict(X_scaled)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

print("Accuracy without splitting:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Step 4: Train the Model Using a Train-Test Split
from sklearn.model_selection import train_test_split

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the logistic regression model on the training set
model_split = LogisticRegression()
model_split.fit(X_train, y_train)

# Predict on the test set
y_test_pred = model_split.predict(X_test)

# Evaluate the model
accuracy_split = accuracy_score(y_test, y_test_pred)
conf_matrix_split = confusion_matrix(y_test, y_test_pred)

print("Accuracy with train-test split:", accuracy_split)
print("Confusion Matrix (test set):\n", conf_matrix_split)


# Accuracy Comparison Visualization
def plot_accuracy_comparison(accuracy_no_split, accuracy_split):
    plt.figure(figsize=(8, 5))
    categories = ["No Split", "Train-Test Split"]
    accuracies = [accuracy_no_split, accuracy_split]

    plt.bar(categories, accuracies, color=["skyblue", "salmon"])
    plt.ylim(0, 1)
    plt.title(
        "Accuracy Comparison Using and Not using Train Test Split in Logistic Regression",
        fontsize=12,
    )
    plt.grid(axis="y", linestyle="--", alpha=1)
    plt.ylabel("Accuracy")
    plt.show()


# Plot Accuracy Comparison
plot_accuracy_comparison(accuracy, accuracy_split)

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Train the Model Without Cross-Validation
model_no_cv = LogisticRegression()
model_no_cv.fit(X_scaled, y)

# Predict on the same dataset
y_pred_no_cv = model_no_cv.predict(X_scaled)

# Evaluate the model without cross-validation
accuracy_no_cv = accuracy_score(y, y_pred_no_cv)
conf_matrix_no_cv = confusion_matrix(y, y_pred_no_cv)

print("Accuracy without cross-validation:", accuracy_no_cv)
print("Confusion Matrix (no CV):\n", conf_matrix_no_cv)

# Step 2: Train the Model Using Cross-Validation
cv_accuracies = cross_val_score(
    model_no_cv, X_scaled, y, cv=5
)  # 5-fold cross-validation
accuracy_cv = np.mean(cv_accuracies)

print("Cross-Validation Accuracy:", accuracy_cv)


# Accuracy Comparison Visualization
def plot_accuracy_comparison(accuracy_no_cv, accuracy_cv):
    plt.figure(figsize=(8, 5))
    categories = ["No Cross-Validation", "Cross-Validation"]
    accuracies = [accuracy_no_cv, accuracy_cv]

    plt.bar(categories, accuracies, color=["skyblue", "salmon"])
    plt.ylim(0, 1)
    plt.title(
        "Accuracy Comparison using Cross Validation and Not Using Cross Validation in Logistic Regression",
        fontsize=12,
    )
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Plot the accuracies
plot_accuracy_comparison(accuracy_no_cv, accuracy_cv)


# Step 2: Train the Model Without Train/Test Split
model_no_split = LinearRegression()
model_no_split.fit(X_scaled, y)

# Predict on the same dataset
y_pred_no_split = model_no_split.predict(X_scaled)

# Evaluate the model without train/test split
mse_no_split = mean_squared_error(y, y_pred_no_split)
r2_no_split = r2_score(y, y_pred_no_split)

print("Linear Regression Mean Squared Error (No Split):", mse_no_split)
print("Linear Regression R^2 Score (No Split):", r2_no_split)

# Step 3: Train the Model Using Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model_split = LinearRegression()
model_split.fit(X_train, y_train)

# Predict on the test set
y_test_pred = model_split.predict(X_test)

# Evaluate the model
mse_split = mean_squared_error(y_test, y_test_pred)
r2_split = r2_score(y_test, y_test_pred)

print("Linear Regression Mean Squared Error (Train-Test Split):", mse_split)
print("Linear Regression R^2 Score (Train-Test Split):", r2_split)

# Step 4: Train the Model Using Cross-Validation
model_cv = LinearRegression()
cv_accuracies = cross_val_score(
    model_cv, X_scaled, y, cv=5, scoring="neg_mean_squared_error"
)
mse_cv = -np.mean(cv_accuracies)  # Convert to positive MSE

# R^2 score is not directly available from cross_val_score, so we calculate it separately
model_cv.fit(X_scaled, y)
r2_cv = r2_score(y, model_cv.predict(X_scaled))

print("Linear Regression Mean Squared Error (Cross-Validation):", mse_cv)
print("Linear Regression R^2 Score (Cross-Validation):", r2_cv)

# Step 5: Train the Model Without Cross-Validation
model_no_cv = LinearRegression()
model_no_cv.fit(X_scaled, y)

# Predict on the same dataset
y_pred_no_cv = model_no_cv.predict(X_scaled)

# Evaluate the model without cross-validation
mse_no_cv = mean_squared_error(y, y_pred_no_cv)
r2_no_cv = r2_score(y, y_pred_no_cv)

print("Linear Regression Mean Squared Error (No Cross-Validation):", mse_no_cv)
print("Linear Regression R^2 Score (No Cross-Validation):", r2_no_cv)


# Accuracy Comparison Visualization
def plot_accuracy_comparison(mse_no_split, mse_split, mse_cv, mse_no_cv):
    categories = [
        "No Split",
        "Train-Test Split",
        "Cross-Validation",
        "No Cross-Validation",
    ]
    mse_values = [mse_no_split, mse_split, mse_cv, mse_no_cv]

    # Normalization for better visualization
    normalized_accuracies = [
        1 - (mse / max(y.mean(), 1 - y.mean())) for mse in mse_values
    ]

    # Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(categories, normalized_accuracies, marker="o", color="b", linestyle="-")
    plt.title("Linear Regression Accuracy Comparison (Line Plot)", fontsize=14)
    plt.ylabel("Normalized Accuracy", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(categories, normalized_accuracies, color="orange", s=100)
    plt.title("Linear Regression Accuracy Comparison (Scatter Plot)", fontsize=14)
    plt.ylabel("Normalized Accuracy", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Plot the accuracies using line and scatter plots
plot_accuracy_comparison(mse_no_split, mse_split, mse_cv, mse_no_cv)
