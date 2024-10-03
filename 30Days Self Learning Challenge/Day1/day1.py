import matplotlib.pyplot as plt
import numpy as np

# Define the equation y = mx + c, where m = 1, c = 1
m = 1
c = 1

# Generate x values
x = np.linspace(0, 2, 100)

# Calculate corresponding y values
y = m * x + c

# Create the plot
plt.figure(figsize=(6,6))
plt.plot(x, y, label='y = mx + c (m=1, c=1)', color='blue')

# Plot the point (1, 2)
plt.scatter(1, 2, color='red', zorder=5)
plt.text(1, 2, '  (1, 2)', fontsize=12)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of y = mx + c (m=1, c=1)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Show the grid
plt.grid(True)

# Show the plot
plt.legend()
plt.show()



# GOOGLE DOCS: https://docs.google.com/document/d/1F4FFsqYd-5EYfnThY_TlYCcTEQ3BQpAvjfIPHBKwGyY/edit?usp=sharing
