import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from cross_validation import *


def plot_confusion_matrix(y, gt):
    cm_d = ConfusionMatrixDisplay(confusion_matrix(y, gt))
    fig, ax = plt.subplots(figsize=(10, 10))
    cm_d.plot(ax=ax)
    
    
def plot_samples(X, y):
    _, axes = plt.subplots(nrows=1, ncols=len(y), figsize=(15, 5))

    for ax, image, label in zip(axes, X, y):
        ax.set_axis_off()
        ax.imshow(image.reshape(28, 28))
        ax.set_title(label)
        
        
def test_classifier(classifier, X_train, y_train, X_test, y_test):
    trained_classifiers, accuracies = cross_validation(classifier, X_train, y_train, 5)
    print(accuracies)
    best_classifier = trained_classifiers[np.argmax(accuracies)]
    
    preds = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print("Best classifier performance: %.4f" %  accuracy)
    
    plot_confusion_matrix(preds, y_test)
    wrong_predictions = [i for i in np.arange(len(preds)) if preds[i] != y_test[i]]
    plot_samples(X_test[wrong_predictions][:8], preds[wrong_predictions][:8])

        