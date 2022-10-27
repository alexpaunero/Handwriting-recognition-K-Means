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

# Task 10
for i in range(10):
 
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
 
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

# Task 11
plt.show()

# Task 15
new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.30,6.56,0.84,0.00,0.00,0.00,0.00,0.00,2.28,7.62,1.52,0.00,0.00,0.00,0.00,0.00,2.28,7.62,1.52,0.00,0.00,0.00,0.00,0.00,2.28,7.62,1.52,0.00,0.00,0.00,0.00,0.00,2.29,7.62,1.52,0.00,0.00,0.00,0.00,0.00,2.06,7.62,2.06,0.00,0.00,0.00,0.00,0.00,0.15,3.51,0.46,0.00,0.00,0.00],
[0.00,0.00,0.00,0.54,0.69,0.00,0.00,0.00,0.00,0.00,0.00,5.03,5.87,0.00,0.00,0.00,0.00,0.00,0.00,5.33,6.10,0.00,0.00,0.00,0.00,0.00,0.00,5.33,6.10,0.00,0.00,0.00,0.00,0.00,0.00,5.34,6.10,0.00,0.00,0.00,0.00,0.00,0.00,5.11,6.79,0.00,0.00,0.00,0.00,0.00,0.00,3.66,7.39,0.00,0.00,0.00,0.00,0.00,0.00,0.23,0.99,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,4.65,4.04,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.33,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.33,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.33,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.34,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.57,0.00,0.00,0.00,0.00,0.00,0.00,3.43,3.51,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,6.79,3.05,0.00,0.00,0.00,0.00,0.00,0.00,7.62,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.62,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.62,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.63,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.55,3.81,0.00,0.00,0.00,0.00,0.00,0.00,2.29,0.76,0.00,0.00,0.00]
])

# Task 16
new_labels = model.predict(new_samples)
print(new_labels)

# Task 17
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')

# Task 18
print(new_labels)
