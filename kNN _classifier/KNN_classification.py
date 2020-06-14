# kNN classifier
import random
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
# ----------------------- K nearest neighbor classification -------------------------#

# The principle behind nearest neighbor methods is to find a predefined number
# of training samples closest in distance to the new point, and predict the label
# from these. The number of samples can be a user-defined constant(k-nearest neighbor learning),
# or vary based on the local density of points(radius-based neighbor learning).
# The distance can, in general, be any metric measure: standard Euclidean distance is the
# most common choice.
# Neighbors-based methods are known as non-generalizing machine learning methods,
# since they simply “remember” all of its training data(possibly transformed into a
#  fast indexing structure such as a Ball Tree or KD Tree).
# Despite its simplicity, nearest neighbors hp
# as been successful in a large number of classification
# and regression problems, including handwritten digits and satellite image scenes.
# Being a non-parametric method, it is often successful in classification situations where
# the decision boundary is very irregular.


# ----------------------- Find euclidean distance from point p to any adjecent point

# Find distance between any two points

def distance(p1, p2):
    """
    Find distance between any two points and return the euclidean distance
    """
    return (np.sqrt(np.sum(np.power(p1-p2, 2))))

# ----------------Find distances for each point from neighbour points---------------------
# We could sort the distances array to give us shorter distances,
# but instead what we really would like to get
# is an index vector that would sort the array.
# If we had that, we could take the first K elements of that array,
# and know that the corresponding points are the K closest
# points to our point of interest p.
# Fortunately, this function exists in NumPy and it's called argsort.
# sort the distance for k pointsthat are nearest


def find_nearest_points(points, p, k):
    """
    Find k nearest points from point p and return their indices.
    """
    distances = np.zeros(points.shape[0])
    for i, point in enumerate(points):
        distances[i] = distance(p, point)
    ind = np.argsort(distances)
    return ind[0:k]

# devide all points based on their class
# Count number of votes for given list


def votes_counts(votes):
    """
    To count all votes and return count for each number of vote.
    """
    votes_counts = {}
    for vote in votes:
        if vote in votes_counts:
            votes_counts[vote] += 1
        else:
            votes_counts[vote] = 1
    return votes_counts

# Find winner for given list of votes


def majority_votes(votes):
    """
    Find which vote has maximum count and return that vote as winner 
    for majority votes.
    If we have more than 1 winner it choose 1 winner at random and give that as winner.
    """
    votes_count = votes_counts(votes)
    max_vote = max(votes_count.values())
    winner = []
    for votes, count in votes_count.items():
        if count == max_vote:
            winner.append(votes)
    return random.choice(winner)
#


def knn_predict(points, p, k, outcomes):
    """
     To find k nearest neghbour and predict the point p belongs to which class
     by checking majority of points belongs to that class for k neighbours and return that class 
     """
    ind = find_nearest_points(points, p, k)
    # predict the class or catagory based on majority votes
    return majority_votes(outcomes[ind])


def generate_synthetic_data(n):
    """
    Generate synthetic data which belongs to class o and class 1 and
    return tuple of points and outcomes.
    """
    points = np.concatenate(
        (ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs((n, 2))), axis=0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)), axis=0)
    return (points, outcomes)
# making grid


def make_prediction_grid(limits, h, points, outcomes, k):
    """
    For each point p in mesh we predict the class it belongs and return 
    x and y coardinate along with class as tuple (x,y,class).
    """
    (xmin, xmax, ymin, ymax) = limits
    xs = np.arange(xmin, xmax, h)
    ys = np.arange(ymin, ymax, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(points, p, k, outcomes)
    return (xx, yy, prediction_grid)


def plot_prediction_grid(xx, yy, prediction_grid):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap(
        ["hotpink", "lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap(["red", "blue", "green"])
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx, yy, prediction_grid,
                   cmap=background_colormap, alpha=0.5)
    plt.scatter(predictors[:, 0], predictors[:, 1],
                c=outcomes, cmap=observation_colormap, s=50)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.xticks(())
    plt.yticks(())
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.savefig(
        "/home/divyesh/PHD2020/python_practice/Python_for_research/kNN _classifier/" + filename)


(predictors, outcomes) = generate_synthetic_data(50)
limits = (-5, 5, -5, 5)
h = 0.1
k = 50
filename = "knn_synth_50.pdf"
(xx, yy, predictors_grid) = make_prediction_grid(
    limits, h, predictors, outcomes, k)
plot_prediction_grid(xx, yy, predictors_grid)

limits = (4, 8, 1.5, 4.5)
h = 0.1
k = 5

# scikit learn
from sklearn import datasets
iris = datasets.load_iris()
#iris

predictors = iris.data[:,0:2]
outcomes = iris.target
plt.plot(predictors[outcomes==0][:,0],predictors[outcomes==0][:,1],"ro")
plt.plot(predictors[outcomes==1][:,0],predictors[outcomes==1][:,1],"go")
plt.plot(predictors[outcomes==2][:,0],predictors[outcomes==2][:,1],"bo")
plt.savefig("irish_data.pdf")


limits = (4, 8, 1.5, 4.5)
h = 0.1
k = 5
filename = "iris_grid_50.pdf"
(xx, yy, predictors_grid) = make_prediction_grid(
    limits, h, predictors, outcomes, k)
plot_prediction_grid(xx, yy, predictors_grid)


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors,outcomes)
sk_prediction = knn.predict(predictors)
#sk_prediction.shape

my_prediction = np.array([knn_predict(predictors,p,5,outcomes) for p in predictors ])
#my_prediction.shape

print (np.mean((sk_prediction ==my_prediction)*100))
print (np.mean((sk_prediction ==outcomes)*100))
print (np.mean((my_prediction ==outcomes)*100))