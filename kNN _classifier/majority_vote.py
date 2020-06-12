def majority_votes(votes):
    """Given list of votes function will 
    count each votes """
    votes_count = {}
    for vote in votes:
        if vote in votes_count:
            votes_count[vote] += 1
        else:
            votes_count[vote] = 1
    return votes_count


votes = [1, 2, 1, 3, 2, 1, 7, 8, 8]
print(majority_votes(votes))
