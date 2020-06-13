# To find K nearest neighbours
import matplotlib.pyplot as plt
import numpy as np
# find distance between point p and all points present in data


def distance(p1, p2):
    """ Find distance between points p1 and p2."""

    return np.sqrt(np.sum(np.power(p1-p2, 2)))


points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [
                  2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
p = np.array([2.5, 2.8])
distances = np.zeros(points.shape[0])

for i in range(len(distances)):
    distances[i] = distance(points[i], p)
print(distances)

plt.plot(points[:, 0], points[:, 1], 'ro')
plt.plot(p[0], p[1], 'bo')
# plt.show()

# we have points and distances corresponding to same place
print(distances[2], points[2])

# We could sort the distances array to give us shorter distances,
# but instead what we really would like to get
# is an index vector that would sort the array.
# If we had that, we could take the first K elements of that array,
# and know that the corresponding points are the K closest
# points to our point of interest p.
# Fortunately, this function exists in NumPy and it's called argsort.
# sort the distance for k pointsthat are nearest

ind = np.argsort(distances)
print(ind)
print(distances[ind])  # distance from point p for each sorted point
print(points[ind])  # we get near points sorted


# if we want tp pick k=3 nearest numbers
k = 3
print(ind[0:k])
print(distances[ind[:k]])
print(points[ind[0:k]])


def find_nearest_points(points, p, k):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(points[i], p)
    ind = np.argsort(distances)
    return (ind[0:k])


test = find_nearest_points(points, p, k)
print(points[test])
