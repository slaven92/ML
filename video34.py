import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

# %%
X = np.array([[1, 2],[1.5, 1.8],[5,8],[8,8],[1,0.6],[9,11]])

# plt.scatter(X[:,0],X[:,1],s=150, linewidths=5)
# plt.show()
# %%
clf = KMeans(n_clusters=6)
clf.fit(X)

centroid = clf.cluster_centers_
labels = clf.labels_

colors = 10*["g.","r.","c.","b.","k.","y."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=20)
plt.scatter(centroid[:,0],centroid[:,1], marker='x', s=150, linewidths=5)
plt.show()
