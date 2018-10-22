import csv
import numpy
import random
import math

#Density,%Fat,Age,Weight,Height,Neck,Chest,Abdom,Hip,Thigh,Knee,Ankle,Biceps,F-arm,Wrist
# 0        1   2    3       4    5     6     7    8    9    10    11   12     13     14


def fit_linear(phi, T):
    T = numpy.array(T, dtype=float)
    phi = numpy.mat(phi, dtype=float)
    return((phi.T * phi).I * (phi.T @ T).T)
    

def test_linear(X_test, T_test, w):
    X_test = numpy.mat(X_test, dtype=float)
    T_Test = numpy.array(T_test, dtype=float)

    t_hat = [w.T @ X_test[i,:].T for i in range(len(T_Test))]
    RMS = 0.0
    for i in range(len(t_hat)):
        RMS += (float(T_test[i]) - float(t_hat[i][0,0]))**2
    RMS /= len(T_Test)
    RMS = math.sqrt(RMS)
    return (RMS)

def radial_basis(x, x_j, j, s):
    if j == 0:
        return(1)
    phi = math.exp(-((x - x_j)**2 / (2 * s**2)))
    return(phi)


def main():
#region Test-Train-Split
    
    with open('bodyfat.csv', 'r') as f:
        reader = list(csv.reader(f, delimiter = ','))[1:]
        M = numpy.array(reader, dtype=float)

    T_temp = M[:,1]
    X_temp = M[:,[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

    N = len(T_temp)
    test_smpls = random.sample(range(0, N), N//10)
    train_smpls = list([i for i in range(N) if i not in test_smpls])

    T_test = T_temp[test_smpls]
    X_test = X_temp[test_smpls,:]

    T_train = T_temp[train_smpls]
    X_train = X_temp[train_smpls,:]
#endregion
#region MultiLinModel
    phi = numpy.ones((X_train.shape[0], X_train.shape[1] + 1))
    phi[:,1:] = X_train
    phi_test = numpy.ones((X_test.shape[0], X_test.shape[1] + 1))
    phi_test[:,1:] = X_test

    w_lin = fit_linear(phi, T_train)
    RMS_lin = test_linear(phi_test, T_test, w_lin)

    print("RMS error for linear model: {:.2f}".format(RMS_lin))
#endregion
#region SingleVarRadialBasis
    #Using abdom
    x_train = X_train[:,6]
    x_test = X_test[:,6]

    M = 3
    x_min = min(x_train)
    x_max = max(x_train)
    s = (x_max - x_min)/(M-1)
    xjs = numpy.linspace(x_min, x_max, M-1)
    
    phi = numpy.zeros((len(x_train), M))

    for i in range(len(x_train)):
        for j in range(M):
            phi[i, j] = radial_basis(x_train[i], xjs[j-1] if j > 0 else 0, j, s)

    phi_test = numpy.zeros((len(x_test), M))

    for i in range(len(x_test)):
        for j in range(M):
            phi_test[i, j] = radial_basis(x_test[i], xjs[j-1] if j > 0 else 0, j, s)

    w_GLR = fit_linear(phi, T_train)
    RMS_GLR = test_linear(phi_test, T_test, w_GLR)
    print("RMS Error for GLR model is: {:.2f}".format(RMS_GLR))
#endregion
#region return
    return (RMS_lin, RMS_GLR)
#endregion



if __name__ == '__main__':
    n = 100
    avg_lin = 0
    avg_GLR = 0
    for i in range(n):
        print("\nRun number {}\n------------------".format(i))
        lin, GLR = main()
        avg_lin += lin
        avg_GLR += GLR
    avg_lin /= n
    avg_GLR /= n
    print("\nAverage for Linear model: {:.2f}\nAverage for GLR model: {:.2f}".format(avg_lin, avg_GLR))