
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import requests
import io
from matplotlib.widgets import Button

# Load the dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
response = requests.get(url)
data = response.content.decode('utf-8')
df = pd.read_csv(io.StringIO(data), header=None)

# Assign column names (from dataset description)
columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
df.columns = columns

# Convert 'Diagnosis' to a numeric target variable and drop 'ID'
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})
df = df.drop(columns=['ID'])

# Initialize Variance Threshold
threshold = 0.5
selector = VarianceThreshold(threshold)

# Track if animation is running
animation_running = False
current_step = 0

# Plot setup
fig, ax = plt.subplots(figsize=(15, 9))
plt.subplots_adjust(bottom=0.2)  # Space for buttons

# Functions for controlling the animation
def start(event):
    global animation_running
    animation_running = True
    animate_feature_selection()

def stop(event):
    global animation_running
    animation_running = False

def reset(event):
    global current_step
    current_step = 0
    animation_running = False
    ax.clear()
    ax.set_title("Feature Selection Animation Reset")
    ax.set_xlabel("Feature Values")
    ax.set_ylabel("Density")
    plt.draw()

# Function to animate feature selection
def animate_feature_selection():
    global current_step
    while animation_running and current_step < len(df.columns) - 1:
        features_to_test = df.drop('Diagnosis', axis=1).iloc[:, :current_step+1]
        selector.fit(features_to_test)
        
        # Selected features based on variance threshold
        selected_features = features_to_test.columns[selector.get_support()]
        
        # Display current step and selected features
        print(f"Step {current_step + 1}: Selected Features: {list(selected_features)}")
        
        # Clear previous plot and plot current features
        ax.clear()
        for feature in df.columns[1:-1]:  # Exclude 'Diagnosis'
            if feature in selected_features:
                sns.histplot(df[feature], kde=True, color="green", label=f"{feature} (Selected)", ax=ax)
            else:
                sns.histplot(df[feature], kde=True, color="red", label=f"{feature} (Eliminated)", ax=ax)
        
        # Title, labels, and legend
        ax.set_title(f"Feature Selection: Step {current_step + 1}")
        ax.set_xlabel("Feature Values")
        ax.set_ylabel("Density")
        
        # Adjust the legend size by setting fontsize
        ax.legend(loc="upper right", fontsize='small', frameon=True)  # Use 'small' for smaller font size
        
        # Display mathematical formula as annotation
        formula_text = (
            r"$\text{Variance of } X = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2$" + "\n"
            r"$\text{Variance Threshold} = 0.5$"
        )
        ax.text(0.02, 1.05, formula_text, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Update plot
        plt.draw()
        plt.pause(1)  # Delay for visualization
        
        # Move to next step
        current_step += 1

# Button setup
ax_start = plt.axes([0.1, 0.05, 0.1, 0.075])  # Position for start button
ax_stop = plt.axes([0.21, 0.05, 0.1, 0.075])  # Position for stop button
ax_reset = plt.axes([0.32, 0.05, 0.1, 0.075])  # Position for reset button

btn_start = Button(ax_start, 'Start')
btn_stop = Button(ax_stop, 'Stop')
btn_reset = Button(ax_reset, 'Reset')

btn_start.on_clicked(start)
btn_stop.on_clicked(stop)
btn_reset.on_clicked(reset)

# Show plot with interactive controls
plt.show()

