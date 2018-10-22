import csv
import numpy
import math
import matplotlib.pyplot as plt
#"Month","Clearwater River at Kamiah, Idaho. 1911 ? 1965"

def radial_basis(x, x_j, s):
    phi = math.exp(-((x - x_j)**2 / (2 * s**2)))
    return(phi)

def Post_Cov(phi, alpha, beta):
    S = (alpha * numpy.identity(phi.shape[1], dtype=float) + beta * phi.T * phi).I
    return S

def Post_Mean(phi, S, beta, T):
    m = beta * S * phi.T @ T
    return m

def Pred_mean(m, phi_X):
    return m @ phi_X

def Pred_stdDev(phi_x, S, beta):
    sigma = 1/beta + phi_x @ S @ phi_x
    return sigma

def main():
    with open('clearwater.csv', 'r') as f:
        #Read in file
        reader = list(csv.reader(f, delimiter = ','))[1:]
        #Make into numpy array.
        data = numpy.array(reader)
    
    #Using int sequence instead of dates
    T = numpy.array(data[:,1], dtype=float)
    N = len(T)
    x = numpy.array([i for i in range(N)])

    #Prep for Radial basis functions
    M = 20
    x_min = min(x)
    x_max = max(x)
    s = (x_max - x_min)/(M-1)
    xjs = numpy.linspace(x_min, x_max, M-1)

    #matrix of ones
    phi = numpy.ones((N, M), dtype=float)

    for i in range(N):
        for j in range(1, M):
            phi[i, j] = radial_basis(x[i], xjs[j-1], s)
    phi = numpy.matrix(phi, dtype=float)
    
    alpha = 0.01
    beta = 0.07

    S_N = Post_Cov(phi, alpha, beta)
    m_N = Post_Mean(phi, S_N, beta, T)
    
    x_test = numpy.linspace(0, N, num = 1000)
    
    m_pred = [float(Pred_mean(m_N, numpy.append(1, numpy.array([radial_basis(x_i, x_j, s) for x_j in xjs])))) for x_i in x_test]
    sigma_pred = [float(Pred_stdDev(numpy.append(1, numpy.array([radial_basis(x_i, x_j, s) for x_j in xjs])), S_N, beta)) for x_i in x_test]

    #plus minus one stdDev
    upper = [sum(x) for x in zip(m_pred, sigma_pred)]
    lower = [i - j for i, j in zip(m_pred, sigma_pred)]

    plt.plot(x_test, m_pred, color = 'red')
    plt.scatter(x, T, color = 'blue')
    plt.fill_between(x_test, lower, upper, facecolor='brown', alpha=0.5)
    plt.show()



if __name__ == '__main__':
    main()