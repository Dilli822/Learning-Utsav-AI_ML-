import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm

# 1. Discrete Random Variables (Bernoulli Distribution)
p = 0.6  # Probability of success (1)
bernoulli_rv = bernoulli(p)

x_discrete = [0, 1]  # Discrete outcomes for Bernoulli
y_discrete = bernoulli_rv.pmf(x_discrete)  # PMF for Bernoulli

# 2. Continuous Random Variables (Normal Distribution)
mu, sigma = 0, 1  # Mean and standard deviation
x_continuous = np.linspace(-3, 3, 1000)  # Continuous range for normal distribution
y_continuous = norm.pdf(x_continuous, mu, sigma)  # PDF for normal distribution

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 1. Plot for Discrete Data (PMF)
ax[0].stem(x_discrete, y_discrete, basefmt=" ", markerfmt="ro", label='PMF (Bernoulli)')
ax[0].set_title("PMF of Bernoulli Distribution")
ax[0].set_xlabel("Random Variable")
ax[0].set_ylabel("Probability")
ax[0].set_xticks([0, 1])
ax[0].legend()

# 2. Plot for Continuous Data (PDF)
ax[1].plot(x_continuous, y_continuous, label='PDF (Normal)', color='blue')
ax[1].set_title("PDF of Normal Distribution")
ax[1].set_xlabel("Random Variable")
ax[1].set_ylabel("Density")
ax[1].legend()

plt.tight_layout()
plt.show()

#　SciPy in Python primarily for its robust collection of mathematical algorithms, statistical functions, and scientific computing tools. 
# .stem() function in Matplotlib creates a stem plot, which is used to display discrete data. 
# A stem plot is similar to a bar chart but with lines (stems) connecting the data points to a baseline. 
# It’s particularly useful for visualizing discrete distributions like PMFs (Probability Mass Functions) or any data with clearly defined, separate points