import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# %%
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],[8,2],[10,2],[9,3]])
colors = 10 * ["g", "r", "c", "b", "k", "y "]
# plt.scatter(X[:,0],X[:,1],s=150, linewidths=5)
# plt.show()
# %%

class Mean_Shift():
    """"""

    def __init__(self, radius = None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []

            for i in centroids:
                in_bandwith = []
                centroid = centroids[i]
                for featureset in data:



                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwith.append(featureset)

                new_centroid = np.average(in_bandwith, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break

        self.centroids = centroids

    def predict(self):
        pass

clf = Mean_Shift(radius = 4)
clf.fit(X)
centroids = clf.centroids
plt.scatter(X[:,0],X[:,1],s=150, linewidths=5)
for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1], marker='*', s=150, color='k')
plt.show()
