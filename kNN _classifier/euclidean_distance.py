import numpy as np


def distance(p1, p2):
    """ Find distance between points p1 and p2."""

    return np.sqrt(np.sum(np.power(p1-p2, 2)))


p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
print(distance(p1, p2))
