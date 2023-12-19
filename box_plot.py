import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility 
np.random.seed(42)

# Generate random data
data1 = np.random.normal(loc=0, scale=1, size=100) 
data2 = np.random.normal(loc=0, scale=2, size=100) 
data3 = np.random.normal(loc=0, scale=0.5, size=100)

# Create a list of data for box plots 
data_to_plot = [data1, data2, data3] 
# print(data_to_plot)

# Create a box plot
plt.boxplot(data_to_plot, vert=True, patch_artist=True)

# Add labels and title
plt.xticks([1, 2, 3], ['Data 1', 'Data 2', 'Data 3']) 
plt.xlabel('Datasets')
plt.ylabel('Values')
plt.title('Box Plot of Random Data')
plt.show()