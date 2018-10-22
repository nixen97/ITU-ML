import math
import numpy as np
import matplotlib.pyplot as plt

#b
def N(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    '''
    Implementation of a bivariate Gaussian distrobution
    '''
    result = 1/(2 * math.pi * sigma1 * sigma2 * math.sqrt(1 - rho**2)) #Normalisation term
    xmmu = np.array([x1 - mu1, x2 - mu2], dtype = float) #x - mu
    epsilon = np.matrix([[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]], dtype=float) #Cov matrix
    result *= math.exp(- 1/2 * xmmu @ epsilon.I @ xmmu) #Exponential term
    return result

#c
X1, X2 = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 100))
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex = 'col', sharey = 'row')

pltList = [ax1, ax2, ax3, ax4, ax5, ax6]
mu1s = [0, 1, 0, 0, 0,       0]
mu2s = [0, 1, 0, 0, 0,       0]
s1s =  [1, 1, 1, 2, 1,       1]
s2s =  [1, 1, 2, 1, 1,       1]
rhos = [0, 0, 0, 0, 0.5,    -0.75]

for i in range(6):
    f = np.vectorize(lambda x1, x2: N(x1, x2, mu1s[i], mu2s[i], s1s[i], s2s[i], rhos[i])    )
    Z = f(X1, X2)
    pltList[i].set_title("(\u03BC\u2081, \u03BC\u2082, \u03C3\u2081, \u03C3\u2082, \u03C1) = ({}, {}, {}, {}, {})".format(mu1s[i], mu2s[i], s1s[i], s2s[i], rhos[i]))
    pltList[i].contour(X1, X2, Z)

plt.subplots_adjust(hspace = 0.35, wspace = 0.45)
plt.show()

#d
#mu 1 and 2:
#Center of distro in x1 and x2
#sigma 1 and 2:
#"Stretch" in  x1 and x2
#rho
#Rotation