# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

# # Step 1: Load the Data
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
# columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
# data = pd.read_csv(url, header=None, names=columns)

# # Step 2: Preprocess the Data
# data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
# X = data.drop(columns=['ID', 'Diagnosis'])
# y = data['Diagnosis']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 3: Train Random Forest and XGBoost Models
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# xgb_model.fit(X_train, y_train)

# # Step 4: Evaluate the Models
# y_pred_rf = rf_model.predict(X_test)
# y_pred_xgb = xgb_model.predict(X_test)

# # Calculate accuracy
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

# # Print performance metrics
# performance_table = pd.DataFrame({
#     'Model': ['Random Forest', 'XGBoost'],
#     'Accuracy': [rf_accuracy, xgb_accuracy]
# })
# print(performance_table)
# print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
# print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# # Step 5: Visualize one Decision Tree from the Random Forest
# plt.figure(figsize=(15, 10))
# plot_tree(rf_model.estimators_[0], feature_names=X.columns, class_names=['Benign', 'Malignant'], filled=True, rounded=True)
# plt.title("Visualization of a Single Decision Tree from Random Forest")
# plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns

# Step 1: Load the Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)

# Step 2: Preprocess the Data
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
X = data.drop(columns=['ID', 'Diagnosis'])
y = data['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest and XGBoost Models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Step 4: Evaluate the Models
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

# Print performance metrics
performance_table = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [rf_accuracy, xgb_accuracy]
})
print(performance_table)
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Step 5: Visualize one Decision Tree from the Random Forest
plt.figure(figsize=(15, 10))
plot_tree(rf_model.estimators_[0], feature_names=X.columns, class_names=['Benign', 'Malignant'], filled=True, rounded=True)
plt.title("Visualization of a Single Decision Tree from Random Forest")
plt.show()

# Step 6: Confusion Matrix for Both Models
confusion_rf = confusion_matrix(y_test, y_pred_rf)
confusion_xgb = confusion_matrix(y_test, y_pred_xgb)

# Plot Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
axes[0].set_title('Random Forest Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
            xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
axes[1].set_title('XGBoost Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Step 7: True vs. Predicted Line Plot
plt.figure(figsize=(10, 6))

# Create a DataFrame for plotting
results_df = pd.DataFrame({
    'True Values': y_test,
    'Random Forest Predictions': y_pred_rf,
    'XGBoost Predictions': y_pred_xgb
})

# Plotting with different line styles
plt.plot(results_df.index, results_df['True Values'], marker='o', linestyle='-', color='black', label='True Values', markersize=5) # Straight line for true values
plt.plot(results_df.index, results_df['Random Forest Predictions'], marker='*', linestyle=':', color='blue', alpha=0.7, label='Random Forest Predictions', markersize=5) # Dotted line for Random Forest
plt.plot(results_df.index, results_df['XGBoost Predictions'], marker='o', linestyle='--', color='red', alpha=0.7, label='XGBoost Predictions', markersize=5) # Dashed line for XGBoost

plt.title('True vs. Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Class (0 = Benign, 1 = Malignant)')
plt.xticks(ticks=np.arange(len(y_test)), labels=np.arange(1, len(y_test)+1))  # Label x-axis with sample numbers
plt.legend()
plt.grid(True)
plt.show()



# Step 7: 3D Scatter Plot for True vs. Predicted Values
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a DataFrame for plotting
results_df = pd.DataFrame({
    'True Values': y_test,
    'Random Forest Predictions': y_pred_rf,
    'XGBoost Predictions': y_pred_xgb
})

# Create an array for sample indices
sample_indices = np.arange(len(y_test))

# Scatter plots
ax.scatter(sample_indices, results_df['True Values'], zs=0, zdir='y', color='black', marker='o', label='True Values')
ax.scatter(sample_indices, results_df['Random Forest Predictions'], zs=1, zdir='y', color='blue', marker='*', alpha=0.7, label='Random Forest Predictions')
ax.scatter(sample_indices, results_df['XGBoost Predictions'], zs=2, zdir='y', color='red', marker='^', alpha=0.7, label='XGBoost Predictions')

# Setting labels
ax.set_xlabel('Sample Index')
ax.set_ylabel('Model')
ax.set_zlabel('Class (0 = Benign, 1 = Malignant)')

# Set y-ticks to show model names
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['True Values', 'Random Forest', 'XGBoost'])

# Set title and legend
ax.set_title('3D Visualization of True vs. Predicted Values')
ax.legend()

plt.show()