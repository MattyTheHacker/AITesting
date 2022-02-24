import numpy as np
from sklearn.cluster import KMeans

centroids = np.array([(3, 0), (5, 7)])
x = np.array([(3, 1), (5, 1), (4, 2), (5, 2), (2, 3), (7, 4), (1, 0), (8, 0)])
model = KMeans(n_clusters=2, init=centroids, n_init=1).fit(x)
print(model.labels_)