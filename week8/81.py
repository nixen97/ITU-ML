#!/usr/bin/env python3

import matplotlib.pyplot as plt

# Read in data
data = []
with open("faithful.csv", "r") as f:
    data = list(map(lambda x: float(x.replace('\n', '')), f))


# Histogram
# Looks like it could be two gaussians
plt.hist(data)
#plt.show()