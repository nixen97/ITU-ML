#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import random

# Math function
def prob1d(x, mean, std):
    frac = 1 / (std * np.sqrt(2 * np.pi))
    power = -0.5 * ((x - mean) / std)**2

    return frac * np.exp(power)

def Pr(x, means, sds, pis, n, k):
    res = 0
    for i in range(n):
        intermres = 0
        for j in range(k):
            temp = pis[j] * prob1d(x[i], means[j], sds[j])
            intermres += 0 if np.isinf(intermres) else temp
        if intermres == 0:
            raise ZeroDivisionError()
        res += np.log(intermres)
    return res

# Printing function
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
fig, (ax1, ax2) = plt.subplots(1, 2)
ax2.hist(data, bins=15, density=True)
#plt.show()

k = 2 # Number of clusters; SHOULD be able to take any value. May not converge though
n = len(data)
max_iter = 1000

# Initialization
# TODO: k-means initialization

means = np.zeros(k, dtype=float)
sds   = np.ones(k, dtype=float)
pis   = np.repeat(1/n, k)
taus  = np.zeros(k, dtype=float)

means = np.array(np.random.randint(1, 6, k), dtype=float)

sds = np.array(np.random.randint(1, 2, k), dtype=float)

# EM Algorithm

# Log likelihood vector
Q = np.zeros(max_iter, dtype=float)

for L in range(max_iter):
    # E step
    respons = np.zeros((k, n))
    
    for j in range(n):
        total_prob = 0
        for i in range(k):
            prob = pis[i] * prob1d(data[j], means[i], sds[i])
            
            respons[i,j] = prob
        
    respons[np.isinf(respons)] = 0.5
    respons = respons / np.sum(respons,axis=0)

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

    #Log likelihood for iteration
    Q[L] = Pr(data, means, sds, pis, n, k)

    if any(np.isnan(means)) or any(np.isnan(sds)):
        exit(0) # We done fucked up

    print("\n*****\n")
    print("L = %d" % (L+1))
    printndarray("Nc's: ", nc)
    printndarray("Pi's: ", pis)
    printndarray("Means: ", means)
    printndarray("Sigmas: ", sds)
    print("Log likelihood: %f" % (Q[L]))
    print("Log likelihood delta: %f" % (np.abs(Q[L-1] - Q[L])))

    if np.abs(Q[L-1] - Q[L]) < 0.000001:
        break


x = np.linspace(1, 5.5, 100)
y = np.sum(np.array(list(map(lambda i: pis[i] * prob1d(x, means[i], sds[i]), range(k)))), axis=0)

y_cum = np.cumsum(y)

ax2.plot(x, y, color='red', linewidth=2)
ax1.plot(x, y_cum, color='blue', linewidth=2)
plt.show()

    