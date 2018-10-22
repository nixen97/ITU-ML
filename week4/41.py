import math
import numpy as np
import matplotlib.pyplot as plt

#(a)

def N(x, mu, sigma2):
    '''
    Implementation of the univariate Gaussian PDF
    '''
    return 1/(math.sqrt(2 * math.pi * sigma2)) * math.exp(-(x - mu)**2 / (2 * sigma2))



#b
xs = np.linspace(-5, 5, 50)

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row')

pltList = [ax1, ax2, ax3]

sigmas = [1, 1, 5]
mus = [0, 3, 0]

for i in range(3):
    pltList[i].set_title("(\u03BC, \u03C3\u00B2) = ({}, {})".format(mus[i], sigmas[i]))
    ys = list([N(x, mus[i], sigmas[i]) for x in xs])
    pltList[i].plot(xs, ys, color = 'blue')

    #c
    x_rand = np.random.normal(mus[i], math.sqrt(sigmas[i]), 10) #Wants stddev instead of variance
    pltList[i].scatter(x_rand, [0 for j in range(10)], color = 'red')

plt.subplots_adjust(hspace=.35)
plt.show()

#d
#They are mean and variance
#As the names imply they are the mean value and the variance or "spread" of the bell curve
