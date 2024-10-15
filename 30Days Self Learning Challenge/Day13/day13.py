import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import csr_matrix

# Step 1: Generate a dense dataset with a curvy relationship
np.random.seed(42)
n_samples = 1000
n_features = 2

X_dense = np.random.randn(n_samples, n_features) * 5
y_classification = (np.sin(X_dense[:, 0]) + np.cos(X_dense[:, 1]) > 0).astype(int)  # Binary target
y_regression = np.sin(X_dense[:, 0]) + np.cos(X_dense[:, 1])  # Continuous target

# Step 2: Introduce sparsity into the dataset (set values < 1 to 0)
X_sparse = X_dense.copy()
X_sparse[X_sparse < 1] = 0
X_sparse_csr = csr_matrix(X_sparse)  # Sparse matrix format

# Step 3: Polynomial transformation (curvy data)
poly = PolynomialFeatures(degree=3)
X_dense_poly = poly.fit_transform(X_dense)
X_sparse_poly = poly.fit_transform(X_sparse_csr.toarray())

# Step 4: Train-test split
X_train_dense, X_test_dense, y_train_class, y_test_class = train_test_split(X_dense_poly, y_classification, test_size=0.2, random_state=42)
X_train_sparse, X_test_sparse, y_train_reg, y_test_reg = train_test_split(X_sparse_poly, y_regression, test_size=0.2, random_state=42)

# Step 5: Classification with Logistic Regression
classifier_dense = LogisticRegression(max_iter=200)  # Increased max_iter
classifier_sparse = LogisticRegression(max_iter=200)

# Train and predict on dense data
classifier_dense.fit(X_train_dense, y_train_class)
y_pred_class_dense = classifier_dense.predict(X_test_dense)

# Train and predict on sparse data
classifier_sparse.fit(X_train_sparse, y_train_class)
y_pred_class_sparse = classifier_sparse.predict(X_test_sparse)

# Step 6: Regression with Linear Regression
regressor_dense = LinearRegression()
regressor_sparse = LinearRegression()

# Train and predict on dense data
regressor_dense.fit(X_train_dense, y_train_reg)
y_pred_reg_dense = regressor_dense.predict(X_test_dense)

# Train and predict on sparse data
regressor_sparse.fit(X_train_sparse, y_train_reg)
y_pred_reg_sparse = regressor_sparse.predict(X_test_sparse)

# Check sizes before plotting
print("Size of X_dense[:, 0]:", X_dense[:, 0].size)
print("Size of y_pred_reg_dense:", y_pred_reg_dense.size)

# Step 8: Scatter plots for regression results
plt.figure(figsize=(12, 6))

# Scatter plot for dense regression
plt.subplot(1, 2, 1)
plt.scatter(X_test_dense[:, 1], y_test_reg, color='blue', label='True Values', alpha=0.6)
plt.scatter(X_test_dense[:, 1], y_pred_reg_dense, color='red', label='Predicted Values', alpha=0.6)
plt.title('Regression (Dense Data)')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.legend(loc='upper right')

# Scatter plot for sparse regression
plt.subplot(1, 2, 2)
plt.scatter(X_test_sparse[:, 1], y_test_reg, color='blue', label='True Values', alpha=0.6)
plt.scatter(X_test_sparse[:, 1], y_pred_reg_sparse, color='red', label='Predicted Values', alpha=0.6)
plt.title('Regression (Sparse Data)')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

# Create a sample dataset
n_samples = 100
continuous_data = np.random.normal(loc=50, scale=10, size=n_samples)
discrete_data = np.random.randint(1, 6, size=n_samples)
categorical_data = np.random.choice(['A', 'B', 'C'], size=n_samples)

# Introduce missing values
missing_rate = 0.2
mask = np.random.random(n_samples) < missing_rate
continuous_data[mask] = np.nan
discrete_data = pd.Series(discrete_data)
discrete_data[mask] = np.nan
categorical_data[mask] = np.nan  # Use np.nan for consistency

# Create a DataFrame
df = pd.DataFrame({
    'Continuous': continuous_data,
    'Discrete': discrete_data,
    'Categorical': categorical_data
})

# Function to plot data before and after imputation
def plot_imputation(ax, original, imputed, title, y_label):
    non_nan = ~pd.isna(original)
    ax.scatter(np.arange(len(original))[non_nan], original[non_nan], c='blue', label='Original', alpha=0.7, s=30)
    ax.scatter(np.arange(len(imputed)), imputed, c='red', label='Imputed', alpha=0.5, s=20)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Sample Index', fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.legend(fontsize=6, loc='best')

# Set up the plot
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('Imputation Visualization', fontsize=14)

# Perform imputations and plot
imputers = {
    'Continuous': SimpleImputer(strategy='mean'),
    'Discrete': SimpleImputer(strategy='most_frequent'),
    'Categorical': SimpleImputer(strategy='most_frequent')
}

for i, (col, imputer) in enumerate(imputers.items()):
    imputed = imputer.fit_transform(df[[col]])
    if col == 'Categorical':
        original = pd.Categorical(df[col]).codes
        imputed = pd.Categorical(imputed.flatten()).codes
    else:
        original = df[col]
    plot_imputation(axes[i], original, imputed.flatten(), f'{col} Data Imputation', 'Value')

# Adjust layout and display
plt.tight_layout()
plt.subplots_adjust(right=0.85, hspace=0.3)

# Add a detailed legend to the figure
fig.legend(
    ['Original Data: Existing values',
     'Imputed Data: Filled-in missing values'],
    loc='center right',
    bbox_to_anchor=(0.98, 0.5),
    fontsize=8,
    title='Legend',
    title_fontsize=10
)

# Display the plot
plt.show()

# Print statistics
for col, imputer in imputers.items():
    original = df[col]
    imputed = imputer.fit_transform(df[[col]]).flatten()
    print(f"\n{col} Data:")
    print(f"Original mean/mode: {original.mean() if col != 'Categorical' else original.mode()[0]}")
    print(f"Imputed mean/mode: {imputed.mean() if col != 'Categorical' else pd.Series(imputed).mode()[0]}")
    
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import csr_matrix

# Step 1: Generate a dense dataset with a curvy relationship
np.random.seed(42)
n_samples = 1000
n_features = 2

X_dense = np.random.randn(n_samples, n_features) * 5
y_classification = (np.sin(X_dense[:, 0]) + np.cos(X_dense[:, 1]) > 0).astype(int)  # Binary target
y_regression = np.sin(X_dense[:, 0]) + np.cos(X_dense[:, 1])  # Continuous target

# Step 2: Introduce sparsity into the dataset (set values < 1 to 0)
X_sparse = X_dense.copy()
X_sparse[X_sparse < 1] = 0
X_sparse_csr = csr_matrix(X_sparse)  # Sparse matrix format

# Step 3: Polynomial transformation (curvy data)
poly = PolynomialFeatures(degree=3)
X_dense_poly = poly.fit_transform(X_dense)
X_sparse_poly = poly.fit_transform(X_sparse_csr.toarray())

# Step 4: Train-test split
X_train_dense, X_test_dense, y_train_class, y_test_class = train_test_split(X_dense_poly, y_classification, test_size=0.2, random_state=42)
X_train_sparse, X_test_sparse, y_train_reg, y_test_reg = train_test_split(X_sparse_poly, y_regression, test_size=0.2, random_state=42)

# Step 5: Visualization of raw input data
plt.figure(figsize=(12, 6))

# Scatter plot for raw dense data
plt.subplot(1, 2, 1)
plt.scatter(X_dense[:, 0], X_dense[:, 1], c=y_classification, cmap='viridis', alpha=0.6)
plt.title('Raw Input Data (Dense)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Target (Classification)')

# Scatter plot for raw sparse data
plt.subplot(1, 2, 2)
plt.scatter(X_sparse[:, 0], X_sparse[:, 1], c=y_classification, cmap='viridis', alpha=0.6)
plt.title('Raw Input Data (Sparse)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Target (Classification)')

plt.tight_layout()
plt.show()

# Step 6: Visualization of transformed data
plt.figure(figsize=(12, 6))

# Scatter plot for transformed dense data
plt.subplot(1, 2, 1)
plt.scatter(X_dense_poly[:, 1], X_dense_poly[:, 2], c=y_classification, cmap='viridis', alpha=0.6)
plt.title('Transformed Input Data (Dense)')
plt.xlabel('Polynomial Feature 1')
plt.ylabel('Polynomial Feature 2')
plt.colorbar(label='Target (Classification)')

# Scatter plot for transformed sparse data
plt.subplot(1, 2, 2)
plt.scatter(X_sparse_poly[:, 1], X_sparse_poly[:, 2], c=y_classification, cmap='viridis', alpha=0.6)
plt.title('Transformed Input Data (Sparse)')
plt.xlabel('Polynomial Feature 1')
plt.ylabel('Polynomial Feature 2')
plt.colorbar(label='Target (Classification)')

plt.tight_layout()
plt.show()
