from pip._internal.utils.misc import enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


class Classifier:

    Model = enum(MLP='MLP', DT='DT', LR='LR', KNN='KNN', GC='GC', RF='RF', RC='RC')

    def __init__(self, model=Model.MLP):
        if model is self.Model.MLP:
            self.model = MLPClassifier(hidden_layer_sizes=(7), learning_rate='adaptive',
                                 solver='adam', alpha=0.5, activation='relu', tol=0.000001, warm_start=True, max_iter=12000000, verbose=True, n_iter_no_change=8000)
        elif model is self.Model.DT:
            self.model = DecisionTreeClassifier()
        elif model is self.Model.LR:
            self.model = LogisticRegression(solver='saga', multi_class='multinomial')
        elif model is self.Model.GC:
            kernel = 1.0 * RBF(1.0)
            self.model = GaussianProcessClassifier(kernel=kernel, random_state=0, multi_class='one_vs_one', max_iter_predict=10000, warm_start=True)
        elif model is self.Model.RF:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        elif model is self.Model.RC:
            self.model = RidgeClassifier()
        else:
            self.model = KNeighborsClassifier(n_neighbors=4, weights='distance', leaf_size=100000, algorithm='kd_tree', n_jobs=100)

    def train(self, data, classification):
        return self.model.fit(data, classification)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def split_training_dataset(data, classification, percentage=0.1):
        return train_test_split(data, classification, test_size=percentage)

