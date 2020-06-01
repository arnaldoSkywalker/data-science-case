from mpl_toolkits.mplot3d import Axes3D
from pip._internal.utils.misc import enum
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import  AgglomerativeClustering
from sklearn import metrics


class Clustering:

    Model = enum(KMEANS='Kmeans', HC='HC', DBSCAN='DBSCAN')

    def __init__(self, data_set, no_clusters, plot_result=True):
        self.data_set = data_set
        self.no_clusters = no_clusters
        self.plot_result = plot_result

    def plot(self, clusters):
        if len(self.data_set.data_points) > 1 and 1 < len(self.data_set.data_points[0]) <= 3:
            if len(self.data_set.data_points[0]) == 2:
                self.__plot2d(clusters)
            else:
                self.__plot3d(clusters)
        else:
            print('Too many dimensions for plotting')

    def __plot2d(self, clusters):
        plt.scatter(self.data_set.data_points[:, 0], self.data_set.data_points[:, 1] if len(
            self.data_set.segmentation_vars) > 1 else self.data_set.data_points[:, 0], c=clusters.labels_,
                    cmap='rainbow')
        plt.yticks(())
        plt.legend()
        plt.show()

    def __plot3d(self, clusters):
        fig = plt.figure(1, figsize=(14, 13))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.scatter(self.data_set.data_points[:, 0], self.data_set.data_points[:, 1], self.data_set.data_points[:, 2],
                   c=clusters.labels_, edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel(self.data_set.segmentation_vars[0])
        ax.set_ylabel(self.data_set.segmentation_vars[1])
        ax.set_zlabel(self.data_set.segmentation_vars[2])
        ax.dist = 12
        plt.show()

    def exec(self, model=Model.KMEANS):
        if model is self.Model.KMEANS:
            self.model = KMeans(n_clusters=self.no_clusters, max_iter=10000)
            self.clusters = self.model.fit(self.data_set.data_points)
        elif model is self.Model.HC:
            self.model = AgglomerativeClustering()
            self.clusters = self.model.fit(self.data_set.data_points)
        else:
            self.model = DBSCAN(eps=0.0905, min_samples=5)
            self.clusters = self.model.fit(self.data_set.data_points)
        if self.plot_result:
            self.plot(self.clusters)
        return self.clusters

    def evaluate_silhouette(self):
        labels = self.model.labels_
        return metrics.silhouette_score(self.data_set.data_points, labels)

    def evaluate_calinski_harabaz_score(self):
        labels = self.model.labels_
        return metrics.calinski_harabaz_score(self.data_set.data_points, labels)

    @staticmethod
    def clustering_to_dicc(clusters):
        i = 0
        dicc = {}
        for c in clusters.labels_:
            if not c in dicc:
                dicc[c] = [i]
            else:
                dicc[c].append(i)
            i += 1
        return dicc

    @staticmethod
    def print_clustering(dicc):
        for key in dicc:
            print("Cluster " + str(key) + str(dicc[key]))