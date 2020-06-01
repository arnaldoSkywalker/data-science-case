# ------------------------------------------------------------------------------------
# Author: Arnaldo Perez Castano
# email: arnaldo.skywalker@gmail.com
# ------------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
from data_science.dataset_manipulation import dataset
from data_science.prediction.classifier import Classifier
from data_science.segmentation.clustering import Clustering

# Main execution data science module
# data_set file location
file = r'../raw_data/case_study.xlsx'

# Decides task to execute (if True executes clustering, otherwise executes prediction)
exec_clustering = True

# General config
start_col_pca = 1
end_col_pca = 8
n_components = 1
apply_pca = True

#segmentation_vars = ['age', 'height', 'openness', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism' ]
#segmentation_vars = ['height', 'conscientiousness', 'neuroticism', 'openness']  kmeans, 4, S = 0.51

segmentation_vars = ['openness', 'age', 'extraversion', 'agreeableness']

# get data_set
data_set = dataset.DataSet(file, segmentation_vars, scale=True)

# ------------------------------------------------------------------------------------
# Customer Segmentation (Clustering)
# ------------------------------------------------------------------------------------
no_clusters = 3

if apply_pca:
    data_set.apply_pca(start_col_pca, end_col_pca, 0, start_col_pca, components=n_components)

if exec_clustering:
    # Execute segmentation
    c_model = Clustering.Model.DBSCAN
    clustering = Clustering(data_set, no_clusters, plot_result=True)
    clusters = clustering.exec(model=c_model)
    dicc = Clustering.clustering_to_dicc(clusters)

    # Print clustering
    Clustering.print_clustering(dicc)

    # Save resulting clustering in CSV with measures such as mean, sum of profiling vars.
    data_set.save(dicc, c_model)

    # Evaluate clustering
    print("Clustering Silhouette Evaluation: %s" % clustering.evaluate_silhouette())

# ------------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------------
else:
    # var to use as output for the classifier (y)
    col = 'item20'
    y = data_set.data[col].astype(int)
    # Divide dataset in training sub-dataset and testing subdataset
    x_train, x_test, y_train, y_test = Classifier.split_training_dataset(data_set.data_points, y, percentage=0.07)

    model = Classifier(model=Classifier.Model.RC)
    model.train(x_train, y_train)
    predicted = model.predict(x_test)
    score = 0

    for i in range(len(x_test)):
        predicted_i = predicted[i]
        correct = y_test.values[i]
        if predicted_i == correct:
            score += 1
        print("X=%s, Predicted=%s, Correct=%s" % (x_test[i], predicted_i, correct))

    print("Mean square error: %s" % mean_squared_error(y_test, predicted))
    print("Score: %s" % str(float(score / len(x_test))))

