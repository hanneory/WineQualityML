import numpy as np

def shuffle(d):
    #rearrange samples
    np.random.shuffle(d)

    #split into groups
    arr = np.array_split(d, 40)
    return arr
