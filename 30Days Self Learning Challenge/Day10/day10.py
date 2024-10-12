import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Define the Data
data = {
    'Hours': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
    'Pass': [0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
             1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Define the dependent and independent variables
X = df['Hours']  # Independent variable
y = df['Pass']   # Dependent variable

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)

# Step 3: Fit the logistic regression model using statsmodels
model = sm.Logit(y, X)  # Logit model
result = model.fit()     # Fit the model

# Print the summary of the regression results
print(result.summary())

# Step 4: Get the coefficients
beta_0 = result.params[0]  # Intercept
beta_1 = result.params[1]  # Coefficient for Hours

# Step 5: Calculate Predicted Probabilities
df['Log-Odds (t)'] = beta_0 + beta_1 * df['Hours']  # Log-Odds
df['Predicted Probability (p)'] = 1 / (1 + np.exp(-df['Log-Odds (t)']))  # Probability

# Step 6: Visualization
plt.figure(figsize=(10, 6))

# Scatter plot for Pass
plt.scatter(df['Hours'], df['Pass'], color='blue', label='Data Points', s=100)

# Plot logistic curve
hours_range = np.linspace(0, 6, 100)  # Range for hours
log_odds = beta_0 + beta_1 * hours_range
probabilities = 1 / (1 + np.exp(-log_odds))  # Logistic function

plt.plot(hours_range, probabilities, color='red', label='Logistic Curve')
plt.title('Logistic Regression: Probability of Passing vs. Hours Studied', fontsize=16)
plt.xlabel('Hours Studied', fontsize=14)
plt.ylabel('Probability of Passing', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axhline(0.5, color='grey', linestyle='--', label='Threshold (p=0.5)')
plt.legend(fontsize=12)
plt.grid()
plt.show()

# Step 7: User Input for Hours Studied
while True:
    user_input = input("Enter the number of hours studied (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        break
    
    try:
        hours_studied = float(user_input)  # Convert input to float
        
        # Calculate predicted probability for the input hours
        log_odds_user = beta_0 + beta_1 * hours_studied
        probability_user = 1 / (1 + np.exp(-log_odds_user))
        
        # Determine pass/fail based on probability (using a threshold of 0.5)
        pass_fail = "Pass" if probability_user >= 0.5 else "Fail"
        
        # Display the result
        print(f"---------Predicted Probability of Passing for {hours_studied} hours studied: {probability_user:.4f} ({pass_fail})-----")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
