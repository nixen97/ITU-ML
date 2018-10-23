#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def dnorm(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2) ))

def prob1d(x, mean, std):
    frac = 1 / (std * np.sqrt(2 * np.pi))
    power = -0.5 * ((x - mean) / std)**2

    return frac * np.exp(power)

def printndarray(text, arr):
    fs = text
    n = len(arr)
    fs += "["
    fs += "%.3f "*n
    fs = fs[:-1] + "]"
    print(fs % tuple(arr))

def sum_not_inf(arr):
    arr = np.array(list(arr))
    return np.nansum(arr[np.isinf(arr) == False])
    

# Read in data
with open("faithful.csv", "r") as f:
    data = np.array(list(map(lambda x: float(x.replace('\n', '')), f)), dtype=float)


# Histogram
# Looks like it could be two gaussians
plt.hist(data, bins=15, density=True)
#plt.show()

k = 2 # Assume two distributions
n = len(data)
it = 20

# Initialization
# TODO: Make this K-means

means = np.zeros(k, dtype=float)
sds   = np.ones(k, dtype=float)
pis   = np.ones(k, dtype=float)
taus  = np.zeros(k, dtype=float)

pis[0] = 0.5
pis[1] = 0.5

means[0] = 1.7
means[1] = 4.4

sds[0] = 0.5
sds[1] = 1.5

#EM Algorithm

#Log likelihood vector
Q = np.zeros(it, dtype=float)

for L in range(it):
    # E step
    respons = np.zeros((k, n))
    
    for j in range(n):
        total_prob = 0
        for i in range(k):
            prob = pis[i] * prob1d(data[j], means[i], sds[i])
            
            respons[i,j] = prob
        
    respons[np.isinf(respons)] = 0.5
    respons = respons / np.sum(respons,axis=0)

    #print(respons)
    #exit(0)
    # M step
    nc = np.zeros(k)
    for i in range(n):
        for j in range(k):
            nc[j] += respons[j,i]
    
    pis = nc/n

    means = (respons @ data) / nc
    
    diff = np.zeros(n)
    for j in range(k):
        for i in range(n):
            diff = data[i] - means[j]
            sds[j] += respons[j,i] * diff**2
    sds /= nc
    sds = np.sqrt(sds)

    print("\n*****\n")
    print("L = %d" % (L+1))
    printndarray("Nc's: ", nc)
    printndarray("Pi's: ", pis)
    printndarray("Means: ", means)
    printndarray("Sigmas: ", sds)

    # TODO: Compute the log likelihood and add an exit condition to exit when
    # log likelihood change is very small.

    if any(np.isnan(means)) or any(np.isnan(sds)):
        exit(0) # We done fucked up


x = np.linspace(1, 5.5, 100)
y = pis[0] * prob1d(x, means[0], sds[0]) + pis[1] * prob1d(x, means[1], sds[1])

plt.plot(x, y, color='red', linewidth=2)
plt.show()

# Compute for L = 0 and L = 1
# Q[0] = 0
# Q[1] = sum_not_inf(map(lambda i: sum_not_inf(pis[i] * (np.log(pis[i]) + np.log(dnorm(data, means[i], sds[i])))), range(k)))

# L = 1

# while np.abs(Q[L]-Q[L-1]) > 0.000001 and L < 1001:
#     # Intermediate value
#     taus = np.array(list(map(lambda i: np.array(pis[i] * dnorm(data, means[i], sds[i]) / \
#         sum(map(lambda i: pis[i] * dnorm(data, means[i], sds[i]), range(k)))), range(k))), dtype=float)

#     taus[np.isinf(taus)] = 0.5

#     tau_sums = np.array(list(map(sum_not_inf, taus)))
    
#     # New pis
#     pis = tau_sums / len(data)
    
#     # New means
#     means = list(map(lambda i: sum_not_inf(taus[i] * data) / tau_sums[i], range(k)))

#     # New SD's
#     sds = np.array(list(map(lambda i: sum_not_inf(taus[i] * (data - means[i])**2) / tau_sums[i], range(k))), dtype=float)

#     # New Log likelihood
#     Q[L+1] = sum(map(lambda i: sum_not_inf(taus[i] * (np.log(pis[i]) + np.log(dnorm(data, means[i], sds[i])))), range(k)))
    
#     # Print some stats
#     print("\n*****\n")
#     print("L = %d" % (L+1))
#     printndarray("Pi's: ", pis)
#     printndarray("Means: ", means)
#     printndarray("Sigmas: ", sds)
#     print("Log likelihood: %f" % (Q[L+1]))
#     print("Delta log likelihood: %f" % (np.abs(Q[L]-Q[L+1])))

#     L += 1
    
    