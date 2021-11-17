import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score


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

    
def search_hyperparams(pipeline, X, y, train_size=5000, test_size=5000):
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size,
                                              test_size=test_size, random_state=0)
    pipeline.fit(X_train, y_train)
    
    best_params = pipeline[-1].best_params_
    best_accuracy = pipeline[-1].best_score_
    
    print(f"best params: { best_params }")
    print(f"best accuracy: { best_accuracy }")
    
    return best_params


def test_classifier(pipeline, X, y, train_size=5000, test_size=5000, metric=accuracy_score):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                        test_size=test_size, random_state=0)
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    accuracy = metric(y_test, preds)
    
    plot_confusion_matrix(preds, y_test)
    
    wrong_predictions = [i for i in np.arange(len(preds)) if preds[i] != y_test[i]]
    plot_samples(X_test[wrong_predictions][:8], preds[wrong_predictions][:8])
    
    print(f"accuracy: { accuracy }")

    
def plot_support_vectors(svc_estimator, vector_shape: tuple, classes_names, num_sampels=7, figsize: tuple=(10, 20)):
    classes_count = svc_estimator.classes_.shape[0]
    n_supports = svc_estimator.n_support_
    vectors = svc_estimator.support_vectors_
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax2 = plt.axes()
    subfigs = fig.subfigures(nrows=classes_count, ncols=1)
    
    for i, (subfig, name) in enumerate(zip(subfigs, classes_names)):
        slice_from = np.sum(n_supports[:i])
        class_vectors = vectors[slice_from: slice_from + num_sampels]
        axs = subfig.subplots(nrows=1, ncols=num_sampels)
        
        subfig.suptitle(f"support vectors for {name}")
        for ax, vector in zip(axs, class_vectors):
            ax.set_axis_off()
            ax.imshow(vector.reshape(vector_shape), interpolation="nearest")
    plt.show()