import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Function to load the diabetes dataset
def load_diabetes_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=column_names)
    return df, ['Glucose', 'BMI', 'Age'], 'Outcome'

# Function to load the wine quality dataset
def load_wine_quality_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    # Convert quality to binary: 1 for good (quality > 5), 0 for bad (quality <= 5)
    df['quality'] = (df['quality'] > 5).astype(int)
    return df, ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'], 'quality'


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the heart disease dataset
def load_heart_disease_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = pd.read_csv(url, header=None, names=column_names, na_values='?')
    df = df.dropna()  # Remove rows with missing values
    
    # Convert 'num' to binary (0 for no disease, 1 for disease)
    df['num'] = (df['num'] > 0).astype(int)
    
    features = ['age', 'cp', 'chol', 'fbs']
    target = 'num'
    
    return df, features, target

# Main function to run the logistic regression
def run_logistic_regression():
    print("Choose a dataset:")
    print("1. Diabetes")
    print("2. Wine Quality")
    print("3. Heart Disease")

    choice = input("Enter the number of your choice (1-3): ")

    if choice == '1':
        df, features, target = load_diabetes_data()
    elif choice == '2':
        df, features, target = load_wine_quality_data()
    elif choice == '3':
        df, features, target = load_heart_disease_data()
    else:
        print("Invalid choice. Exiting.")
        return

    # Prepare the Data
    X = df[features]
    y = df[target]

    # Add a constant to the model (for the intercept)
    X = sm.add_constant(X)

    # Fit the logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()

    # Print the summary of the regression results
    print(result.summary())

    # Calculate Predicted Probabilities
    df['Predicted Probability'] = result.predict(X)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=features[0], y='Predicted Probability', color='blue', label='Predicted Probability')
    plt.title(f'Predicted Probability vs. {features[0]}', fontsize=16)
    plt.xlabel(features[0], fontsize=14)
    plt.ylabel('Predicted Probability', fontsize=14)
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold (p=0.5)')
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    # User Input for Prediction
    if choice == '3':  # If Heart Disease is selected
        while True:
            try:
                age = float(input("Enter age: "))
                cp = float(input("Enter chest pain type (0-3): "))  # Assuming user enters valid type
                chol = float(input("Enter serum cholesterol (mg/dl): "))
                fbs = float(input("Enter fasting blood sugar (1 = > 120 mg/dl, 0 = < 120 mg/dl): "))

                # Create input values for prediction
                input_values = [age, cp, chol, fbs]  # Use only relevant features for heart disease
                log_odds_user = result.params[0] + np.dot(result.params[1:], input_values)
                probability_user = 1 / (1 + np.exp(-log_odds_user))

                # Determine pass/fail based on probability (using a threshold of 0.5)
                pass_fail = "Positive" if probability_user >= 0.5 else "Negative"
                
                # Display the result
                print(f"Predicted Probability: {probability_user:.4f} ({pass_fail})")
            except ValueError:
                print("Invalid input. Please enter numeric values.")

            cont = input("Do you want to make another prediction? (yes/no): ").strip().lower()
            if cont != 'yes':
                break
    else:
        while True:
            user_input = input(f"Enter {features[0]} value (or type 'exit' to quit): ")
            
            if user_input.lower() == 'exit':
                break
            
            try:
                # Create a list of input values, initializing other features to 0
                input_values = [float(user_input)] + [0] * (len(features) - 1)  
                log_odds_user = result.params[0] + np.dot(result.params[1:], input_values)
                probability_user = 1 / (1 + np.exp(-log_odds_user))

                # Determine pass/fail based on probability (using a threshold of 0.5)
                pass_fail = "Positive" if probability_user >= 0.5 else "Negative"
                
                # Display the result
                print(f"Predicted Probability: {probability_user:.4f} ({pass_fail})")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

# Run the logistic regression program
run_logistic_regression()
