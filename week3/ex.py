import numpy
import matplotlib.pyplot as plt
import random
import math

print('\n')
# -- 1 --
X = numpy.loadtxt('x.txt')
Y = numpy.loadtxt('y_1ofK.txt')

N = -1
if len(X) == len(Y):
    N = len(X)
else:
    raise ValueError()

#Show some of the data
for i in range(10):
    s1 = '[{:.2f}, {:.2f}]'.format(X[i,0], X[i,1])
    s2 = '<{:.0f}, {:.0f}, {:.0f}>'.format(Y[i,0], Y[i,1], Y[i,2 ])
    l = len(s1) + len(s2)
    s = s1 + ' ' + '-'*(40-l) + '> ' + s2
    print(s)

# -- 2 --
plt.scatter(X[:,0], X[:,1], color = list(['blue' if Y[i, 0] else 'red' if Y[i, 1] else 'green' for i in range(N)]))
#plt.show()

# -- 3 --
test_smpls = random.sample(range(0, N), N//5)
train_smpls = list([i for i in range(N) if i not in test_smpls])

X_test = X[test_smpls]
Y_test = Y[test_smpls]
N_test = len(X_test)

X_train = X[train_smpls]
Y_train = Y[train_smpls]
N_train = len(X_train)

#plt.scatter(X_train[:,0], X_train[:,1], color='blue')
#plt.scatter(X_test[:,0], X_test[:,1], color='red')
#plt.show()

#Verify split
print("\n> Test set is {:.2f}% of total data\n".format(N_test/N * 100))

# -- 4 --
#Defining a common cov matrix
alpha = .01
epsilon = (1/alpha) * numpy.matrix(numpy.identity(2))

t_ovr = [0]*N_train
#Getting means according to max likelyhood
for i in range(N_train):
    if Y_train[i,0] == 1:
        t_ovr[i] = 1
t_ovr = numpy.array(t_ovr)

t_ovr_test = [0]*N_test
for i in range(N_test):
    if Y_test[i,0] == 1:
        t_ovr_test[i] = 1
t_ovr_test = numpy.array(t_ovr_test)

N_1 = sum(t_ovr)
N_2 = N_train - N_1

mu_1 = 1/N_1 * numpy.array([sum(x) for x in zip(*[t_ovr[n] * X_train[n,:] for n in range(N_train)])])

mu_2 = 1/N_2 * numpy.array([sum(x) for x in zip(*[(1 - t_ovr[n]) * X_train[n,:] for n in range(N_train)])])

#Compute w
w = epsilon.I @ (mu_1 - mu_2)
w_0 = float(-(1/2) * mu_1 @ epsilon.I @ mu_1 + (1/2) * mu_2 @ epsilon.I @ mu_2 + math.log((N_1/N_train)/(N_2/N_train)))

#Some definitions
def sigmoid(a):
    return 1/(1 + math.exp(-a))

def classify(x, w, w_0):
    p_1 = sigmoid(w @ x + w_0)
    return 1 if p_1 > 0.5 else 0

#Compute train/test error
train_error = 0
for i in range(N_train):
    if classify(X_train[i,:], w, w_0) != t_ovr[i]:
        train_error += 1

test_error = 0
for i in range(N_test):
    if classify(X_test[i,:], w, w_0) != t_ovr_test[i]:
        test_error += 1

print("Barolo or not Barolo; that is the question\n" + '-'*50 + '\n' + "Training error:\t{:.2f}%\nTest error:\t{:.2f}%".format(train_error/N_train*100, test_error/N_test*100) + '\n' + '-'*50 + '\n')

# -- 5 --
def a(x, w_k, w_k0):
    return w_k @ x + w_k0

def softMax(a_1, a_2, a_3):
    result = [0]*3
    result[0] = math.exp(a_1)/(sum([math.exp(a_1), math.exp(a_2), math.exp(a_3)]))
    result[1] = math.exp(a_2)/(sum([math.exp(a_1), math.exp(a_2), math.exp(a_3)]))
    result[2] = math.exp(a_3)/(sum([math.exp(a_1), math.exp(a_2), math.exp(a_3)]))
    return result

N_1 = sum(Y_train[:,0])
N_2 = sum(Y_train[:,1])
N_3 = sum(Y_train[:,2])

mu_1 = 1/N_1 * numpy.array([sum(x) for x in zip(*[X_train[n,:] for n in range(N_train) if Y[n,0] == 1])])
mu_2 = 1/N_2 * numpy.array([sum(x) for x in zip(*[X_train[n,:] for n in range(N_train) if Y[n,1] == 1])])
mu_3 = 1/N_3 * numpy.array([sum(x) for x in zip(*[X_train[n,:] for n in range(N_train) if Y[n,2] == 1])])

w_1 = epsilon.I @ mu_1
w_2 = epsilon.I @ mu_2
w_3 = epsilon.I @ mu_3

w_10 = float(-(1/2) * mu_1 @ epsilon.I @ mu_1 + math.log(N_1/N_train))
w_20 = float(-(1/2) * mu_2 @ epsilon.I @ mu_2 + math.log(N_2/N_train))
w_30 = float(-(1/2) * mu_3 @ epsilon.I @ mu_3 + math.log(N_3/N_train))


for i in range(N_train):
    res = softMax(a(X_train[i], w_1, w_10), a(X_train[i], w_2, w_20), a(X_train[i], w_3, w_30))
    #print(res)
