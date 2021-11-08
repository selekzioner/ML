import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def cross_validation(classifier, X, y, folds):
    k_fold = KFold(n_splits=folds, shuffle=False)
    
    trained_classifiers = []
    accuracies = []

    for train_index, val_index in k_fold.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        classifier.fit(X_train_fold, y_train_fold)
        accuracy = accuracy_score(y_val_fold, classifier.predict(X_val_fold))

        trained_classifiers.append(classifier)
        accuracies.append(accuracy)

    return trained_classifiers, accuracies

