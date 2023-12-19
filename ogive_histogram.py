import matplotlib.pyplot as plt 
import numpy as np

# Sample data - each value represents a bin and it's number of occurrences the frequency
data = [1.6, 2.1, 4.2, 8.6, 9.6, 1.5, 2.7, 4.6, 10, 10.4, 1.2, 2.3, 5.2, 10.5, 11.8, 1.4, 2.5, 5.4, 10.6, 12.3, 1.6, 2.8,
6.1, 10.8, 11.8, 1.2, 2.9, 6.5, 10.3, 12.5, 1.6, 2.8, 7.6, 9.6, 12.4, 1.6, 2.9, 8.3, 9.1, 11.8, ]

# HISTOGRAM
plt.hist(data, bins=6, edgecolor='black') 
plt.xlabel('Value') 
plt.ylabel('Frequency') 
plt.title('Histogram')
plt.show()

# data = [65, 72, 78, 82, 88, 90, 94, 98, 100, 105, 110, 115, 120, 125] 
# Create a box plot
plt.boxplot(data)
plt.xlabel('Value')
plt.title('Box Plot') 
plt.show()

# Create a histogram to obtain the frequency data
hist, bin_edges = np.histogram(data, bins=6, range=(min(data), max(data))) 
print(hist)
print(bin_edges)

# Calculate midpoints of each bin
bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2

# FREQUENCY POLYGON 
plt.plot(bin_midpoints, hist, marker='o') 
plt.xlabel('Value') 
plt.ylabel('Frequency') 
plt.title('Frequency Polygon') 
plt.show()
# Calculate cumulative frequencies 
cumulative_freq = np.cumsum(hist)

# Plot the OGIVE
plt.plot(bin_edges[:-1], cumulative_freq, marker='o', linestyle='-') 
plt.xlabel('Value')
plt.ylabel('Cumulative Frequency')
plt.title('Ogive')
plt.show()