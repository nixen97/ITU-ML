import numpy as np
import math

A_np = np.array([[1, 2], [3, 4]])
A = [1, 2, 3, 4]
n = 2

def dot(a, b):
    if (len(a) != len(b)):
        raise ValueError('a and b must be same length') 
    return sum([a[i] * b[i] for i in range(len(a))])

def MVMultiply(M, v):
    pass

def MMMultiply(A, B, n):
    if (len(A) != len(B)):
        raise ValueError('A and B must be same size')
    C = [0]*len(A)
    for i in range(n):
        for j in range(n):
            C[i * n + j] = sum([A[i * n + k] * B[k * n + j] for k in range(n)])
    return C

def Identity(n):
    A = [0.0]*(n*n)
    for i in range(n):
        A[i * n + i] = 1.0
    return A

def QRDecomp(A, n):
    '''
    A should be a 1-dimensional array,
    representing a row-major n x n matrix
    '''
    P = Identity(n)
    Q = Identity(n)
    R = A.copy()

    for i in range(n):
        u = [0.0]*n
        v = [0.0]*n

        for j in range(i, n):
            u[j] = R[j * n + i]
        
        alpha = len(u) if u[i] < 0 else -len(u)

        for j in range(n):
            v[j] = u[j] + alpha if j == i else u[j]

        if (len(v) < 0.0001):
            continue
        
        v_norm = math.sqrt(sum([v[w]**2 for w in range(len(v))]))

        for w in range(len(v)):
            v[w] /= v_norm

        dv2 = 2 * dot(v, v)
        print(dv2)
        for j in range(n):
            P[j * n + j] -= dv2
        print(P)
        
        R = MMMultiply(P, R, n)
        Q = MMMultiply(Q, P, n)
    return (Q, R)





Q_np, R_np = np.linalg.qr(A_np)
Q, R = QRDecomp(A, n)

print('Numpy')
print(Q_np, R_np)

print('Manual')
print(Q, R)

