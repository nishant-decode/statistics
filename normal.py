import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm 

def normal_dist(x, mean, sd):
    y = sd * np.sqrt(2 * np.pi)
    z= 1 /y
    prob_density = z * np.exp(-0.5 * ((x - mean) / sd) ** 2) 
    return prob_density

def plot_normal_distribution(mu, sigma):
    x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000) 
    y = norm.pdf(x_range, mu, sigma)
    plt.plot(x_range, y, label=f'Mean={mu}, StdDev={sigma}')

# Values for mean (μ) and standard deviation (σ) mu_values = [0, 1]
sigma_values = [0.5, 1, 1.5]

# Plot normal distributions for various μ and σ for mu in mu_values:
for sigma in sigma_values:
    x = np.random.normal(mu, sigma, 10000) 
    result = normal_dist(x, mu, sigma) 
    plot_normal_distribution(mu, sigma)

# Add labels and title
plt.xlabel('Value')

# Show legend plt.legend()
plt.ylabel('Probability') 

# Show the plot plt.show()
plt.title('Probability Distribution Normal')

