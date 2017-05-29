from sklearn.datasets import fetch_20newsgroups
groups = fetch_20newsgroups()

groups.keys()
groups['target_names']
groups.target
import numpy as np
np.unique(groups.target)

groups.data[0]
groups.target[0]

groups.target_names[groups.target[0]]

len(groups.data[0])
len(groups.data[1])
