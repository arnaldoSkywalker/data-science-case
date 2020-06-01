import pandas as pd
import csv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, Imputer
import locale
locale.setlocale(locale.LC_ALL, 'deu_deu')
from sklearn.decomposition import PCA
import numpy as np
from sklearn.impute import SimpleImputer


class DataSet:

    items = ["item" + str(i) for i in range(1, 31)]
    behaviors = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism' , 'openness']
    other_features = ['usr', 'clothing size', 'age', 'height' ]
    columns_to_remove = other_features + behaviors + items
    mapping = {'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5}

    def __init__(self, file, segmentation_vars, scale=False):
        self.data = pd.read_excel(file, na_values=[""])
        self.data_clone = pd.read_excel(file)
        self.segmentation_vars = segmentation_vars
        # Preprocessing operations on dataset (replace missing values, scale data, handle categorical variables)
        self.__get_data_points()
        self.__replace_missing_values()
        if scale:
            self.__scale_data_points()
        else:
            self.__normalize_data_points()

    def __replace_missing_values(self):
        self.data = self.data.fillna(self.data.mean())

    def __get_data_points(self):
        self.__convert_to_float()
        self.__handle_ordinal_vars('clothing size', self.mapping)
        self.__remove_columns()

    def __scale_data_points(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_points = scaler.fit_transform(self.data_points)

    def __normalize_data_points(self):
        self.data_points = preprocessing.normalize(self.data_points)

    def __remove_columns(self):
        self.data_y = self.data_clone.drop((self.other_features + self.behaviors), 1, inplace=False)
        self.data_clone.drop(list(set(self.columns_to_remove) - set(self.segmentation_vars)), 1, inplace=True)
        self.data_points = self.data_clone.values.tolist()

    def __convert_to_float(self):
        self.data_clone.extraversion = self.data_clone.extraversion.map(lambda s: float(str(s).replace(',', '.')))
        self.data_clone.agreeableness = self.data_clone.agreeableness.map(lambda s: float(str(s).replace(',', '.')))
        self.data_clone.conscientiousness = self.data_clone.conscientiousness.map(lambda s: float(str(s).replace(',', '.')))
        self.data_clone.neuroticism = self.data_clone.neuroticism.map(lambda s: float(str(s).replace(',', '.')))
        self.data_clone.openness = self.data_clone.openness.map(lambda s: float(str(s).replace(',', '.')))

    def __handle_ordinal_vars(self, col, mapping={}):
        if len(mapping) > 0:
            self.data_clone[col] = self.data_clone[col].map(mapping)

    def __units_sold_row(self, elem):
        result = 0
        for col in self.items:
            value = float(self.data.iloc[elem][col])
            result += value
        return result

    def __behavior_per_row(self, elem):
        result = []
        for col in self.behaviors:
            result.append(float(str(self.data.iloc[elem][col]).replace(',', '.')))
        return result

    def ___units_per_item(self, elem):
        result = []
        for col in self.items:
            result.append(self.data.iloc[elem][col])
        return result

    def apply_pca(self, i, j, l, m, components=2):
        x = self.data_points[:, i:j]
        pca = PCA(n_components=components)
        principal_components = pca.fit_transform(x)
        self.data_points = self.data_points[:, l:m]
        self.data_points = np.concatenate((self.data_points, principal_components), axis=1)

    def save(self, clusters_dicc, model):
        with open('clusters_file.csv', mode='w') as clusters_file:
            clusters_writer = csv.writer(clusters_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            clusters_writer.writerow(['Segmentation vars: '] + self.segmentation_vars)
            clusters_writer.writerow(['Model: '] + [model])
            # Save clusters in file
            for key in clusters_dicc:
                cluster_elems = clusters_dicc[key]
                clusters_writer.writerow('Cluster ' + str(key))
                result = 0
                units_per_item = [0]*len(self.items)
                behavior_person = [0] * len(self.items)
                for elem in cluster_elems:
                    units_sold_row = self.__units_sold_row(elem)
                    result += units_sold_row
                    clusters_writer.writerow(self.data.values.tolist()[elem])
                    units_per_item_row = self.___units_per_item(elem)
                    behavior_per_row = self.__behavior_per_row(elem)
                    units_per_item = [x + y for x, y in zip(units_per_item, units_per_item_row)]
                    behavior_person = [x + y for x, y in zip(behavior_person, behavior_per_row)]
                clusters_writer.writerow(['Total rating: ' + str(result)])
                clusters_writer.writerow(['Avg units rating per item: ' +  str([x / len(cluster_elems) for x in units_per_item])])
                clusters_writer.writerow(
                    ['Avg behavior per person: ' + str([x / len(cluster_elems) for x in behavior_person])])

    @staticmethod
    def integer_encoder(data):
        # integer encode
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(data)

    @staticmethod
    def one_hot_enconder(data):
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        return onehot_encoder.fit_transform(integer_encoded)

    @staticmethod
    def equals(l1, l2):
        res = [0 if l1[i] == l2[i] else 1 for i in range(l1.size)]
        return sum(res) == 0