import scipy.stats as ss
import random


def majority_votes(votes):
    """returns most common votes"""
    votes_counts = {}
    for vote in votes:
        if vote in votes_counts:
            votes_counts[vote] += 1
        else:
            votes_counts[vote] = 1
# Findout which vote is repeating maximum time
    max_counts = max(votes_counts.values())

    # in order to find which key associated with max value and to findout winner
    winner = []
    for votes, count in votes_counts.items():
        if count == max_counts:
            winner.append(votes)
        #print(votes, count)
    # using random will give us one winner at random if we have more than 1 winner
    return (random.choice(winner))


votes = [1, 2, 1, 3, 2, 1, 7, 8, 8, 8]
winner = majority_votes(votes)
print(winner)


# we can achive all this with help of using mode in numpy but this will not help us when we have more than 1 winner


def majority_votes_short(votes):
    """
    returns most common votes using mode in votes.
    Return an array of the modal (most common) value in the passed array.
    If there is more than one such value, only the smallest is returned. The bin-count for the modal bins is also returned.
    returns,
    mode: ndarray
        Array of modal values.
    count: ndarray

        Array of counts for each mode.
    """
    mode, count = ss.stats.mode(votes)
    return (mode)


votes = [1, 2, 1, 3, 2, 1, 7, 8, 8, 8]
winner2 = majority_votes_short(votes)
print(winner2)
