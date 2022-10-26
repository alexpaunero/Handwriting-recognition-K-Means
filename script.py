import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

# Task 1
digits = datasets.load_digits()
print(digits)

# Task 2
print(digits.DESCR)

# Task 3
print(digits.data)

# Task 4
print(digits.target)

# Task 5
plt.gray() 
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])

# Task 6
from sklearn.cluster import KMeans

# Task 7
model = KMeans(n_clusters=10, random_state=42)

# Task 8
model.fit(digits.data)

# Task 9
fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
