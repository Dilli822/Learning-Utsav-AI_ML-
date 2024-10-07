import numpy as np
import matplotlib.pyplot as plt

# Function to format the plot titles consistently
def format_title(ax, title, pad=15):
    ax.set_title(title, fontsize=12, loc='center', pad=pad)

# Function to add index to the bottom of each plot
def add_index(fig, ax, index, pad=10):
    fig.text(0.5, 0.01, f"Index: {index}", ha='center', fontsize=12)

# Function to demonstrate conditional probability
def conditional_probability():
    # Simulate rolling two dice
    dice1 = np.random.randint(1, 7, 1000)
    dice2 = np.random.randint(1, 7, 1000)

    # Calculate P(dice1 = 6 | dice2 > 3)
    condition = dice2 > 3
    event = dice1 == 6
    conditional_prob = np.sum(event & condition) / np.sum(condition)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.hist2d(dice1, dice2, bins=6, range=[[1, 7], [1, 7]], cmap='Blues', alpha=0.7)
    plt.axvline(x=6, color='red', linestyle='--', label='Dice 1 = 6')
    format_title(ax, 'Conditional Probability\nDice Rolls', pad=15)
    plt.xlabel('Dice 1', fontsize=10)
    plt.ylabel('Dice 2', fontsize=10)
    plt.colorbar(label='Counts')
    plt.legend()
    plt.grid()
    
    # Add index
    add_index(fig, ax, 1)
    plt.show()
    
    return conditional_prob

# Function to demonstrate chain rule
def chain_rule():
    A = np.random.binomial(1, 0.5, 1000)
    B = np.random.binomial(1, 0.7 * A + 0.3 * (1 - A), 1000)
    C = np.random.binomial(1, 0.6 * B + 0.2 * (1 - B), 1000)

    # Calculate probabilities
    chain_prob = (np.mean(A) * np.mean(B) * np.mean(C))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.scatter(A, B, alpha=0.5)
    format_title(ax, 'Chain Rule\nDependent Events', pad=15)
    plt.xlabel('Event A', fontsize=10)
    plt.ylabel('Event B', fontsize=10)
    plt.grid()
    
    # Add index
    add_index(fig, ax, 2)
    plt.show()

    return chain_prob

# Function to test independence
def independence_test():
    X = np.random.normal(0, 1, 1000)
    Y = np.random.normal(0, 1, 1000)

    # Calculate correlation
    correlation = np.corrcoef(X, Y)[0, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.scatter(X, Y, alpha=0.5)
    format_title(ax, 'Independence Test\nUncorrelated Normal Variables', pad=15)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.grid()
    plt.axis('equal')
    
    # Add index
    add_index(fig, ax, 3)
    plt.show()

    return correlation

# Function to demonstrate variance and covariance
def variance_covariance():
    x = np.random.normal(0, 1, 1000)
    y = 0.7 * x + np.random.normal(0, 0.5, 1000)

    # Calculate variance and covariance
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.cov(x, y)[0, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.5)
    format_title(ax, 'Variance-Covariance\nCorrelated Variables', pad=15)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.grid()
    plt.axis('equal')
    
    # Add index
    add_index(fig, ax, 4)
    plt.show()

    return var_x, var_y, cov_xy

# Function to demonstrate Bernoulli covariance
def bernoulli_covariance(p1=0.6, p2=0.4, p12=0.3):
    X = np.random.binomial(1, p1, 1000)
    p_y_given_x1 = p12 / p1
    Y = np.zeros(1000)
    for i in range(1000):
        if X[i] == 1:
            Y[i] = np.random.binomial(1, p_y_given_x1)
        else:
            p_y_given_x0 = (p2 - p12) / (1 - p1)
            Y[i] = np.random.binomial(1, p_y_given_x0)

    # Calculate covariance
    cov = np.mean(X * Y) - np.mean(X) * np.mean(Y)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.scatter(X, Y, alpha=0.5)
    format_title(ax, 'Bernoulli Covariance', pad=15)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.grid()
    plt.axis('equal')
    
    # Add index
    add_index(fig, ax, 5)
    plt.show()

    return cov

# Function to demonstrate linearity of expectation
def expectation_linearity():
    X = np.random.normal(2, 1, 1000)
    Y = np.random.normal(3, 1, 1000)
    
    a, b = 2, 3
    direct_expectation = np.mean(a * X + b * Y)
    linear_combination = a * np.mean(X) + b * np.mean(Y)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.scatter(X, Y, alpha=0.5)
    format_title(ax, 'Linearity of Expectation', pad=15)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.grid()
    plt.axis('equal')
    
    # Add index
    add_index(fig, ax, 6)
    plt.show()

    return direct_expectation, linear_combination

# Run demonstrations
print("Conditional Probability:", conditional_probability())
print("Chain Rule Probability:", chain_rule())
print("Independence Correlation:", independence_test())
var_x, var_y, cov_xy = variance_covariance()
print(f"Variance X: {var_x:.4f}, Variance Y: {var_y:.4f}, Covariance: {cov_xy:.4f}")
print("Bernoulli Covariance:", bernoulli_covariance())
direct, linear = expectation_linearity()
print(f"Expectation Direct: {direct:.4f}, Linear Combination: {linear:.4f}")

#  ----------------------------------------------------------------------------

#  -------------------------------------- 3D PLOTS --------------------------------------
#  ----------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt

class ProbabilityStats:
    def __init__(self):
        np.random.seed(42)

    def conditional_probability(self, ax, sample_size=1000):
        """Demonstrate conditional probability with dice rolls"""
        dice1 = np.random.randint(1, 7, sample_size)
        dice2 = np.random.randint(1, 7, sample_size)

        condition = dice2 > 3
        event = dice1 == 6
        conditional_prob = np.sum(event & condition) / np.sum(condition)

        hist, xedges, yedges = np.histogram2d(dice1, dice2, bins=6, range=[[1,7], [1,7]])
        x, y = np.meshgrid(xedges[:-1], yedges[:-1])
        ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(hist).ravel(),
                 dx=1, dy=1, dz=hist.ravel(), alpha=0.5)
        ax.set_title('Conditional Probability\nDice Rolls')
        ax.set_xlabel('Dice 1')
        ax.set_ylabel('Dice 2')
        return conditional_prob

    def chain_rule(self, ax, sample_size=1000):
        """Demonstrate chain rule with three events"""
        A = np.random.binomial(1, 0.5, sample_size)
        B = np.random.binomial(1, 0.7 * A + 0.3 * (1 - A), sample_size)
        C = np.random.binomial(1, 0.6 * B + 0.2 * (1 - B), sample_size)

        chain_prob = (np.mean(A) * 
                      np.sum(B[A==1])/np.sum(A) * 
                      np.sum(C[B==1])/np.sum(B))
        
        ax.scatter(A, B, C, alpha=0.1)
        ax.set_title('Chain Rule\nDependent Events')
        ax.set_xlabel('Event A')
        ax.set_ylabel('Event B')
        ax.set_zlabel('Event C')
        return chain_prob

    def independence_test(self, ax, sample_size=1000):
        """Test and visualize independence of random variables"""
        X = np.random.normal(0, 1, sample_size)
        Y = np.random.normal(0, 1, sample_size)

        correlation = np.corrcoef(X, Y)[0, 1]

        hist, xedges, yedges = np.histogram2d(X, Y, bins=20)
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
        ax.plot_surface(xpos, ypos, hist, cmap='viridis', alpha=0.7)
        ax.set_title('Independence Test\nUncorrelated Normal Variables')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return correlation

    def variance_covariance(self, ax, sample_size=1000):
        """Demonstrate variance and covariance"""
        x = np.random.normal(0, 1, sample_size)
        y = 0.7 * x + np.random.normal(0, 0.5, sample_size)

        var_x = np.var(x)
        var_y = np.var(y)
        cov_xy = np.cov(x, y)[0, 1]

        ax.scatter(x, y, np.zeros_like(x), c=x + y, cmap='viridis')
        ax.set_title('Variance-Covariance\nCorrelated Variables')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return var_x, var_y, cov_xy

    def bernoulli_covariance(self, ax, p1=0.6, p2=0.4, p12=0.3, sample_size=1000):
        """Calculate and visualize Bernoulli covariance"""
        X = np.random.binomial(1, p1, sample_size)
        p_y_given_x1 = p12 / p1
        Y = np.zeros(sample_size)
        for i in range(sample_size):
            if X[i] == 1:
                Y[i] = np.random.binomial(1, p_y_given_x1)
            else:
                p_y_given_x0 = (p2 - p12) / (1 - p1)
                Y[i] = np.random.binomial(1, p_y_given_x0)

        cov = np.mean(X * Y) - np.mean(X) * np.mean(Y)

        ax.scatter(X + np.random.normal(0, 0.05, sample_size),
                   Y + np.random.normal(0, 0.05, sample_size),
                   np.zeros_like(X),
                   alpha=0.1)
        ax.set_title('Bernoulli Covariance')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return cov

    def expectation_linearity(self, ax, sample_size=1000):
        """Demonstrate linearity of expectation"""
        X = np.random.normal(2, 1, sample_size)
        Y = np.random.normal(3, 1, sample_size)

        a, b = 2, 3
        direct_expectation = np.mean(a * X + b * Y)
        linear_combination = a * np.mean(X) + b * np.mean(Y)

        ax.scatter(X, Y, a * X + b * Y, alpha=0.1)
        ax.set_title('Linearity of Expectation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('aX + bY')
        return direct_expectation, linear_combination

    def display_results(self):
        """Run all demonstrations and display results"""
        # First figure: Chain Rule, Conditional Probability, Independence Test
        fig1 = plt.figure(figsize=(15, 5))

        ax1 = fig1.add_subplot(131, projection='3d')
        conditional_prob = self.conditional_probability(ax1)

        ax2 = fig1.add_subplot(132, projection='3d')
        chain_prob = self.chain_rule(ax2)

        ax3 = fig1.add_subplot(133, projection='3d')
        correlation = self.independence_test(ax3)

        plt.tight_layout()
        plt.show()

        # Second figure: Variance-Covariance, Bernoulli Covariance, Linearity of Expectation
        fig2 = plt.figure(figsize=(15, 5))

        ax4 = fig2.add_subplot(131, projection='3d')
        var_x, var_y, cov_xy = self.variance_covariance(ax4)

        ax5 = fig2.add_subplot(132, projection='3d')
        bernoulli_cov = self.bernoulli_covariance(ax5)

        ax6 = fig2.add_subplot(133, projection='3d')
        direct, linear = self.expectation_linearity(ax6)

        plt.tight_layout()
        plt.show()

        # Print results
        print(f"Conditional Probability: {conditional_prob:.4f}")
        print(f"Chain Rule Probability: {chain_prob:.4f}")
        print(f"Independence Correlation: {correlation:.4f}")
        print(f"Variance X: {var_x:.4f}, Variance Y: {var_y:.4f}, Covariance: {cov_xy:.4f}")
        print(f"Bernoulli Covariance: {bernoulli_cov:.4f}")
        print(f"Expectation Direct: {direct:.4f}, Linear Combination: {linear:.4f}")

# Run demonstrations
ps = ProbabilityStats()
ps.display_results()
