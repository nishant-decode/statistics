import numpy as np
import math
import scipy.special
import matplotlib.pyplot as plt

def exp_dist(nx, lam): 
    nz = []
    for i in range(len(nx)): 
        nz.append(-lam * i)
    pq = np.zeros(len(nx))
    pq = np.array(nz)
    prob_density = lam * np.exp(pq) 
    return prob_density

lam = 0.5
nx = np.arange(0, 10, 1) 
# print(nx)

result = exp_dist(nx, lam) 
# print(result)

fig, axs = plt.subplots(2, 3)
axs[0, 0].set_title('lam=0.5') 
axs[0, 0].scatter(nx, result) 
axs[0, 0].set_xlabel('nx')
axs[0, 0].set_ylabel('Probability')

lam = 1.0
result = exp_dist(nx, lam)
axs[0, 1].set_title('lam=1')
axs[0, 1].scatter(nx, result) 
axs[0, 1].set_xlabel('nx')
axs[0, 1].set_ylabel('Probability')

lam = 1.5
result = exp_dist(nx, lam) 
axs[0, 2].set_title('lam=1.5')
axs[0, 2].scatter(nx, result) 
axs[0, 2].set_xlabel('nx')
axs[0, 2].set_ylabel('Probability')

lam = 2.0
result = exp_dist(nx, lam)
axs[1, 0].set_title('lam=2.0') 
axs[1, 0].scatter(nx, result)
axs[1, 0].set_xlabel('nx')
axs[1, 0].set_ylabel('Probability')

lam = 2.5
result = exp_dist(nx, lam)
axs[1, 1].set_title('lam=2.5') 
axs[1, 1].scatter(nx, result)
axs[1, 1].set_xlabel('nx')
axs[1, 1].set_ylabel('Probability')

lam = 3.0
result = exp_dist(nx, lam)
axs[1, 2].set_title('lam=3.0') 
axs[1, 2].scatter(nx, result) 
axs[1, 2].set_xlabel('nx')
axs[1, 2].set_ylabel('Probability')
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.6, wspace=0.8)
plt.show()