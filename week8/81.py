#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def Pr():
    pass


# Read in data
data = []
with open("faithful.csv", "r") as f:
    data = list(map(lambda x: float(x.replace('\n', '')), f))


# Histogram
# Looks like it could be two gaussians
plt.hist(data)
#plt.show()

k = 2 # Assume two distributions

# Initialization
# TODO: Make this K-means

means = [0] * k
sds   = [1] * k
pis   = [10] * k

means[0] = 1.32
means[1] = 0.32


#EM Algorithm
#Init Q




