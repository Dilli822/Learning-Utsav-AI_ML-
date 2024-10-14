import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# USE SPARSE VS NON SPARSE LOGISTIC RGESSION

# Load the Breast Cancer Wisconsin dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = [
    'ID', 'Diagnosis', 'Radius_Mean', 'Texture_Mean', 'Perimeter_Mean',
    'Area_Mean', 'Smoothness_Mean', 'Compactness_Mean', 'Concavity_Mean',
    'Concave_Points_Mean', 'Symmetry_Mean', 'Fractal_Dimension_Mean',
    'Radius_Se', 'Texture_Se', 'Perimeter_Se', 'Area_Se', 'Smoothness_Se',
    'Compactness_Se', 'Concavity_Se', 'Concave_Points_Se', 'Symmetry_Se',
    'Fractal_Dimension_Se', 'Radius_Worst', 'Texture_Worst', 'Perimeter_Worst',
    'Area_Worst', 'Smoothness_Worst', 'Compactness_Worst', 'Concavity_Worst',
    'Concave_Points_Worst', 'Symmetry_Worst', 'Fractal_Dimension_Worst'
]
breast_cancer_data = pd.read_csv(url, header=None, names=column_names)

# Drop the ID column
breast_cancer_data = breast_cancer_data.drop(columns=['ID'])

# Encode categorical labels (classification)
le = LabelEncoder()
breast_cancer_data['Diagnosis_encoded'] = le.fit_transform(breast_cancer_data['Diagnosis'])

# Select relevant features for classification and regression
X = breast_cancer_data.drop(columns=['Diagnosis', 'Diagnosis_encoded'])
y_class = breast_cancer_data['Diagnosis_encoded']  # For classification
y_reg = breast_cancer_data['Area_Mean']  # For regression (predicting Area Mean)

# Split into training and test sets
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression (Normal)
logistic_model = LogisticRegression(penalty=None, max_iter=1000)  # Corrected to None
logistic_model.fit(X_train_scaled, y_class_train)
y_log_pred = logistic_model.predict(X_test_scaled)

# 2. Sparse Logistic Regression (L1 regularization)
sparse_logistic_model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)
sparse_logistic_model.fit(X_train_scaled, y_class_train)
y_sparse_log_pred = sparse_logistic_model.predict(X_test_scaled)

# Accuracy comparison (classification)
log_acc = accuracy_score(y_class_test, y_log_pred)
sparse_log_acc = accuracy_score(y_class_test, y_sparse_log_pred)

# 3. Linear Regression (Normal)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_reg_train)
y_lin_pred = linear_model.predict(X_test_scaled)

# 4. Sparse Linear Regression (Lasso)
sparse_linear_model = Lasso(alpha=0.1)  # Regularization strength can be tuned
sparse_linear_model.fit(X_train_scaled, y_reg_train)
y_sparse_lin_pred = sparse_linear_model.predict(X_test_scaled)

# Error comparison (regression)
linear_mse = mean_squared_error(y_reg_test, y_lin_pred)
sparse_linear_mse = mean_squared_error(y_reg_test, y_sparse_lin_pred)

# Set the color palette for clarity
palette = {0: 'blue', 1: 'red'}

# Plotting: 2D Plots for Logistic Regression
plt.figure(figsize=(14, 6))

# 2D plot for Logistic Regression Predictions
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test['Radius_Mean'], y=X_test['Texture_Mean'], 
                hue=y_log_pred, palette=palette, alpha=0.7, legend=False)  # Hide default legend
plt.title('Logistic Regression Predictions')
plt.xlabel('Radius Mean (Average size of the tumor in mm)')
plt.ylabel('Texture Mean (Average texture of the tumor)')
plt.axhline(0.5, color='red', linestyle='--')  # Threshold line

# Custom legend for Logistic Regression
for label in palette.keys():
    plt.scatter([], [], color=palette[label], label=f'{label}: {"Benign" if label == 0 else "Malignant"}')
plt.legend(title='Diagnosis', loc='upper right')

# 2D plot for Sparse Logistic Regression Predictions
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test['Radius_Mean'], y=X_test['Texture_Mean'], 
                hue=y_sparse_log_pred, palette=palette, alpha=0.7, legend=False)  # Hide default legend
plt.title('Sparse Logistic Regression Predictions (L1)')
plt.xlabel('Radius Mean (Average size of the tumor in mm)')
plt.ylabel('Texture Mean (Average texture of the tumor)')
plt.axhline(0.5, color='red', linestyle='--')  # Threshold line

# Custom legend for Sparse Logistic Regression
for label in palette.keys():
    plt.scatter([], [], color=palette[label], label=f'{label}: {"Benign" if label == 0 else "Malignant"}')
plt.legend(title='Diagnosis', loc='upper right')

plt.tight_layout()
plt.show()

# Plotting: 3D Plots for Logistic Regression
fig = plt.figure(figsize=(14, 6))

# 3D plot for Normal Logistic Regression
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test['Radius_Mean'], X_test['Texture_Mean'], y_class_test, 
            c=[palette[pred] for pred in y_log_pred], marker='o', alpha=0.7)
ax1.set_title('3D Logistic Regression (Normal)')
ax1.set_xlabel('Radius Mean (Average tumor size in mm)')
ax1.set_ylabel('Texture Mean (Average texture of the tumor)')
ax1.set_zlabel('Diagnosis (0: Benign, 1: Malignant)')
ax1.view_init(30, 210)

# 3D plot for Sparse Logistic Regression
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test['Radius_Mean'], X_test['Texture_Mean'], y_class_test, 
            c=[palette[pred] for pred in y_sparse_log_pred], marker='o', alpha=0.7)
ax2.set_title('3D Sparse Logistic Regression (L1)')
ax2.set_xlabel('Radius Mean (Average tumor size in mm)')
ax2.set_ylabel('Texture Mean (Average texture of the tumor)')
ax2.set_zlabel('Diagnosis (0: Benign, 1: Malignant)')
ax2.view_init(30, 210)

plt.tight_layout()
plt.show()

#  Plotting: 2D Plots for Linear Regression
plt.figure(figsize=(14, 6))

# 2D plot for Linear Regression Predictions
plt.subplot(1, 2, 1)
plt.scatter(y_reg_test, y_lin_pred, color='blue', alpha=0.7)
plt.title('Linear Regression Predictions')
plt.xlabel('True Values (Area Mean, mm²)')
plt.ylabel('Predicted Values (Area Mean, mm²)')
plt.axline((0, 0), slope=1, color='red', linestyle='--')  # Perfect prediction line

# 2D plot for Sparse Linear Regression Predictions
plt.subplot(1, 2, 2)
plt.scatter(y_reg_test, y_sparse_lin_pred, color='orange', alpha=0.7)
plt.title('Sparse Linear Regression Predictions (Lasso)')
plt.xlabel('True Values (Area Mean, mm²)')
plt.ylabel('Predicted Values (Area Mean, mm²)')
plt.axline((0, 0), slope=1, color='red', linestyle='--')  # Perfect prediction line

plt.tight_layout()
plt.show()

# Plotting: 3D Linear Regression
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111, projection='3d')

# Assuming 'Radius_Mean' is the feature used for the 3D plot
ax.scatter(X_test['Radius_Mean'], y_reg_test, y_lin_pred, color='blue', alpha=0.7)
ax.set_title('3D Linear Regression')
ax.set_xlabel('Radius Mean (Average tumor size in mm)')
ax.set_ylabel('True Values (Area Mean, mm²)')
ax.set_zlabel('Predicted Values (Area Mean, mm²)')

# Optionally, add a plane for better visualization
# Calculate the plane's Z values using a linear equation (for simplicity)
# Replace coefficients with those from your linear regression model
slope = 1  # Adjust slope as needed based on your regression coefficients
intercept = 0  # Adjust intercept as needed
X_range = np.linspace(X_test['Radius_Mean'].min(), X_test['Radius_Mean'].max(), 10)
Y_range = slope * X_range + intercept
X_plane, Y_plane = np.meshgrid(X_range, Y_range)
Z_plane = slope * X_plane + intercept

# Plotting the regression plane
ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='orange')

plt.show()

# Print accuracy and error metrics
print(f"Logistic Regression Accuracy (Normal): {log_acc:.2f}")
print(f"Sparse Logistic Regression Accuracy (L1): {sparse_log_acc:.2f}")
print(f"Linear Regression MSE (Normal): {linear_mse:.2f}")
print(f"Sparse Linear Regression MSE (Lasso): {sparse_linear_mse:.2f}")

# Summary of Predictions
print("\nPredictions Summary:")
print("Logistic Regression Predictions (0: Benign, 1: Malignant):")
print(y_log_pred)
print("\nLinear Regression Predictions (Area Mean):")
print(y_lin_pred)

# Explanations for Logistic and Linear Regression
print("\nExplanations:")
print("Logistic Regression predicts the probability of a tumor being malignant (1) or benign (0) based on features like radius and texture.")
print("Linear Regression predicts continuous outcomes (like area mean) based on input features. A perfect prediction aligns true and predicted values.")