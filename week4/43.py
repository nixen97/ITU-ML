def h(a):
    return max([0, a])

def y(x1, x2):
    return h(x1 + x2) - 2 * h(x1 - 1 + x2 - 1)

for x1 in range(0, 2):
    for x2 in range(0, 2):
        print("({}, {}) => {}".format(x1, x2, y(x1, x2)))