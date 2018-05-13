"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  .. _LFW: http://vis-www.cs.umass.edu/lfw/

  original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html

"""

print (__doc__)

from time import time
import logging
import pylab as pl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

# for machine learning we use the data directly (as relative pixel
# position info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print ("Total dataset size:")
print ("n_samples: %d" % n_samples)
print ("n_features: %d" % n_features)
print ("n_classes: %d" % n_classes)


###############################################################################
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# creating list to measure the changes in f1-score
n_comp_list = [10, 15, 25, 50, 100, 250]

for i in n_comp_list:
    print ("Extracting the top %d eigenfaces from %d faces" % (i, X_train.shape[0]))
    t0 = time()

    pca = PCA(svd_solver='randomized', n_components=i, whiten=True).fit(X_train)
    print ("for n_component value : ", i)
    print ("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((i, h, w))

    print ("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print ("done in %0.3fs" % (time() - t0))

    # Showing the variance ratio:
    print ("\nShowing the explained variance ratio: ")
    print (pca.explained_variance_ratio_)
    # showing the first two variance ratios as required in the question (ANS 1)
    print("\nThe first variance ratio: ", pca.explained_variance_ratio_[0])
    print("The second variance ratio: ", pca.explained_variance_ratio_[1])
    ###############################################################################
    # Train a SVM classification model

    print ("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {
             'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              }
    # for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print ("done in %0.3fs" % (time() - t0))
    print ("Best estimator found by grid search:")
    print (clf.best_estimator_)


    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print ("Predicting the people names on the testing set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print ("done in %0.3fs" % (time() - t0))

    print (classification_report(y_test, y_pred, target_names=target_names))
    print (confusion_matrix(y_test, y_pred, labels=range(n_classes)))



    ## plotting code removed because of its tendancy to break, at n_component value = 10


    ##############################
    # noting down results for the n_components list (the 3rd columns denotes the f1-score)
    #
    # for n_component = 10
    # Ariel Sharon       0.10      0.15      0.12        13
    #
    # for n_component = 15
    # Ariel Sharon       0.23      0.46      0.31        13
    #
    # for n_component = 25
    # Ariel Sharon       0.56      0.69      0.62        13
    #
    # for n_component = 50
    # Ariel Sharon       0.67      0.77      0.71        13
    #
    # for n_component = 100
    # Ariel Sharon       0.71      0.77      0.74        13
    #
    # for n_component = 250
    # Ariel Sharon       0.53      0.77      0.62        13
    ##############################

    ## Inference:

    ## It can be seen that the f1-score increases with the increase in the number of features at first but
    ## after a certain time, with increase in more features, it tends to fall.
