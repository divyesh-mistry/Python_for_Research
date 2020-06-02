
text = "hello, my name is Divyesh and divyesh can do anything."
text = text.lower()
sp_char = [",", ".", ":", ";", "?", "!"]
for ch in sp_char:
    text = text.replace(ch, "")
print(text)


def word_count(text):
    # spit each word from text string and access each word and looping over it
    """
    Count the number of time each word occure in text(str) .
    return dictionary of words where keys are of unique word
    and values are the number of time it has appeared in the string
    """
    word_counts = {}
    for word in text.split(" "):
        # repeated word
        if word in word_counts:
            word_counts[word] += 1
        # new word
        else:
            word_counts[word] = 1
    return word_counts


# we have module available in python wich count words
from collections import Counter

def word_count_fast(text):
    """
    Count the number of time each word occure in text(str) using python module counter object.
    return dictionary of words where keys are of unique word
    and values are the number of time it has appeared in the string
    """
    word_count=Counter(text.split(" "))
    return word_count

print(word_count_fast(text))

