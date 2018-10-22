import numpy
import matplotlib.pyplot as plt


x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

t = [0.15, -0.16, -0.61, -0.86, -1.02, -0.44, 0.16, 0.05, 0.45, 1.39, 0.86]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

pltList = [ax1, ax2, ax3, ax4]

Ml = [1, 2, 5, 10]

for l in range(len(Ml)):

    M = Ml[l] + 1

    T = [0.0]*M

    A = [0.0]*(M*M)

    A_np = numpy.zeros((M,M))

    for i in range(M):

        for n in range(len(x)):

            T[i] += (x[n])**i * t[n]

            for j in range(M):

                A[j * M + i] += (x[n])**(i+j)

                A_np[i, j] = A[j * M + i]

    
    w_np = numpy.linalg.solve(A_np, T)

    xs = numpy.linspace(0, 1, 1000)

    ys = [0]*len(xs)

    ys = [sum([w_np[j] * xs[i]**j for j in range(M)]) for i in range(len(xs))]

    pltList[l].set_title('M = {}'.format(M - 1))

    pltList[l].plot(xs, ys, color = 'red')

    pltList[l].scatter(x, t, color = 'blue')

plt.show()