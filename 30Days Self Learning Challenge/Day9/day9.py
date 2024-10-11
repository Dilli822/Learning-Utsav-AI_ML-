import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generate synthetic data for regression
np.random.seed(0)
X = np.random.randn(100, 1) * 10
y = 2 * X.squeeze() + 5 + np.random.randn(100) * 5

# Create and fit models
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=0.1)
lasso.fit(X, y)
ridge.fit(X, y)

# Create a range of values for prediction
X_range = np.linspace(-30, 30, 300).reshape(-1, 1)

# Predictions
y_lasso = lasso.predict(X_range)
y_ridge = ridge.predict(X_range)

# 2D Plot for Lasso and Ridge
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='lightgray', label='Data Points')
plt.plot(X_range, y_lasso, color='blue', label='Lasso (L1)')
plt.plot(X_range, y_ridge, color='red', label='Ridge (L2)')
plt.title('L1 vs L2 Regularization (2D)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 3D Plot for Lasso and Ridge
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(X, y, np.zeros_like(y), color='lightgray', label='Data Points')
ax.plot(X_range, y_lasso, zs=0, color='blue', label='Lasso (L1)')
ax.plot(X_range, y_ridge, zs=0, color='red', label='Ridge (L2)')
ax.set_title('L1 vs L2 Regularization (3D)')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_zlabel('Regularization Type')
ax.legend()
plt.show()

# Generate synthetic data for K-Means Clustering
X_clustering = np.random.rand(100, 2) * 100
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_clustering)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 2D Plot for K-Means
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_clustering[:, 0], X_clustering[:, 1], c=labels, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', label='Centroids', s=200)
plt.title('K-Means Clustering (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# 3D Plot for K-Means
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(X_clustering[:, 0], X_clustering[:, 1], np.zeros_like(X_clustering[:, 0]), c=labels, cmap='viridis', label='Data Points')
ax.scatter(centroids[:, 0], centroids[:, 1], np.zeros_like(centroids[:, 0]), color='red', label='Centroids', s=200)
ax.set_title('K-Means Clustering (3D)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Cluster')
ax.legend()
plt.show()

# Generate synthetic data for K-Nearest Neighbors
X_knn = np.random.rand(100, 2) * 100
y_knn = np.random.randint(0, 2, 100)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_knn, y_knn)

# Create grid for decision boundary
x_min, x_max = X_knn[:, 0].min() - 1, X_knn[:, 0].max() + 1
y_min, y_max = X_knn[:, 1].min() - 1, X_knn[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

# Predict on the grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 2D Plot for K-Nearest Neighbors
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_knn[:, 0], X_knn[:, 1], c=y_knn, edgecolor='k', marker='o', label='Data Points')
plt.title('K-Nearest Neighbors (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# 3D Plot for K-Nearest Neighbors
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(X_knn[:, 0], X_knn[:, 1], y_knn, c=y_knn, cmap='viridis', label='Data Points')
ax.set_title('K-Nearest Neighbors (3D)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Class')
ax.legend()
plt.show()

# Generate synthetic classification data for cross-validation
X_cv, y_cv = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=0)
model_cv = LogisticRegression()
scores = cross_val_score(model_cv, X_cv, y_cv, cv=5)

# 2D Plot for Cross-Validation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_cv[:, 0], X_cv[:, 1], c=y_cv, cmap='viridis', label='Data Points')
plt.title('Cross-Validation (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# 3D Plot for Cross-Validation
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(X_cv[:, 0], X_cv[:, 1], y_cv, c=y_cv, cmap='viridis', label='Data Points')
ax.set_title('Cross-Validation (3D)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Class')
ax.legend()
plt.show()

print(f"Cross-Validation Scores: {scores}")

# Generate synthetic data for SVM
X_svm = np.random.rand(100, 2) * 100
y_svm = np.random.randint(0, 2, 100)
svm_model = SVC(kernel='linear')
svm_model.fit(X_svm, y_svm)

# Create grid for decision boundary
xx_svm, yy_svm = np.meshgrid(np.arange(X_svm[:, 0].min() - 1, X_svm[:, 0].max() + 1, 1),
                             np.arange(X_svm[:, 1].min() - 1, X_svm[:, 1].max() + 1, 1))

# Predict on the grid
Z_svm = svm_model.predict(np.c_[xx_svm.ravel(), yy_svm.ravel()])
Z_svm = Z_svm.reshape(xx_svm.shape)

# 2D Plot for Support Vector Machine
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx_svm, yy_svm, Z_svm, alpha=0.3, cmap='viridis')
plt.scatter(X_svm[:, 0], X_svm[:, 1], c=y_svm, edgecolor='k', marker='o', label='Data Points')
plt.title('Support Vector Machine (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# 3D Plot for Support Vector Machine
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(X_svm[:, 0], X_svm[:, 1], y_svm, c=y_svm, cmap='viridis', label='Data Points')
ax.set_title('Support Vector Machine (3D)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Class')
ax.legend()
plt.show()

# Generate data for convex and non-convex functions
x_convex = np.linspace(-5, 5, 100)
y_convex = x_convex ** 2

x_non_convex = np.linspace(-5, 5, 100)
y_non_convex = np.sin(x_non_convex)

# 2D Plot for Convex and Non-Convex Functions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_convex, y_convex, color='green', label='Convex Function')
plt.title('Convex Function')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_non_convex, y_non_convex, color='purple', label='Non-Convex Function')
plt.title('Non-Convex Function')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.grid(True)

plt.tight_layout()
plt.show()
