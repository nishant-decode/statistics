import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import binom 

ns = [20,30]

ps = [0.2, 0.5, 0.8] 
plt.figure(figsize=(12, 8)) 
plot_index = 1

for n in ns:
    for p in ps:
        x = np.arange(0, n+1)
        pmf = binom.pmf(x, n, p) 
        plt.subplot(len(ns), len(ps), plot_index) 
        plt.plot(x, pmf, label=f'n={n}, p={p}') 
        plt.title(f'n={n}, p={p}')
        plt.xlabel('X (Number of Successes)') 
        plt.ylabel('Probability')
        plt.legend()
        plot_index += 1

plt.tight_layout() 
plt.show()