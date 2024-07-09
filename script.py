import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load the digits dataset
digits = datasets.load_digits()
print(digits.DESCR)

# Print the digit data (features)
print(digits.data)
target = digits.data
print(target)

# Display a digit image
plt.gray()
plt.matshow(digits.images[100])
plt.show()

# Print the label of the displayed digit image
print(digits.target[100])

# Create and fit the KMeans model
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

# Visualize the cluster centers
fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

# Define new samples
new_samples = np.array([
[0.00,0.00,2.06,6.25,7.62,7.17,0.23,0.00,0.00,2.82,7.40,6.79,7.62,4.12,0.00,0.00,2.90,7.55,5.72,2.52,7.62,1.91,0.00,0.00,7.62,7.02,0.54,2.90,7.62,1.07,0.00,0.00,7.32,7.62,7.47,7.24,7.62,4.96,3.59,0.00,0.00,1.60,4.27,7.62,6.71,7.32,7.17,0.30,0.00,0.00,3.13,7.62,1.68,0.00,0.00,0.00,0.00,0.00,3.66,7.32,0.76,0.00,0.00,0.00],
[0.00,0.00,0.00,0.23,5.41,3.89,0.00,0.00,0.00,0.00,0.53,5.95,7.62,5.34,0.00,0.00,0.00,0.00,6.03,7.47,7.40,5.34,0.00,0.00,0.00,0.00,6.25,2.75,6.10,5.34,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.34,0.00,0.00,0.00,0.00,0.00,0.00,6.10,5.34,0.00,0.00,0.00,0.00,2.75,5.34,7.17,5.87,2.29,1.37,0.00,0.00,4.88,6.86,7.17,7.62,7.62,5.26],
[0.00,0.00,2.59,7.62,3.05,0.00,0.00,0.00,0.00,2.75,7.40,7.62,3.13,0.00,0.00,0.00,0.00,4.88,5.80,7.32,3.81,0.00,0.00,0.00,0.00,0.00,0.00,6.86,4.35,0.00,0.00,0.00,0.00,0.00,0.00,5.87,5.41,0.00,0.00,0.00,0.00,0.00,0.00,4.80,6.48,0.00,0.00,0.00,0.00,0.08,1.45,5.03,7.24,4.27,5.11,5.26,0.00,1.98,7.62,7.62,7.55,6.79,6.10,5.95],
[1.30,7.47,5.80,4.96,4.58,3.28,0.00,0.00,0.23,4.12,5.41,6.10,7.40,7.62,1.37,0.00,0.00,0.00,0.00,2.75,7.55,6.02,0.31,0.00,0.00,0.00,1.98,7.55,7.47,1.37,0.00,0.00,0.92,1.75,5.41,7.47,7.63,4.27,0.00,0.00,4.73,7.62,2.44,2.98,6.71,6.10,0.00,0.00,1.68,7.40,7.62,7.62,7.47,3.81,0.00,0.00,0.00,1.60,3.05,2.59,0.61,0.00,0.00,0.00]
])

# Predict the labels of the new samples
new_labels = model.predict(new_samples)
print(new_labels)

# Map the predicted labels to their respective digit representations
label_map = {
    0: 0,
    1: 9,
    2: 2,
    3: 1,
    4: 6,
    5: 8,
    6: 4,
    7: 5,
    8: 7,
    9: 3
}

# Print the mapped digit predictions
for label in new_labels:
    print(label_map[label], end='')
