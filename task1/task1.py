import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class PotentialFunctionClassifier:
    def __init__(self, window_size: float):
        self.window_size = window_size
        
    def _calculate_kernel(self, x: np.array):
        return 1 / (x + 1)

    def fit(self, train_x: np.array, train_y: np.array, epochs: int):
        self.train_x = train_x
        self.train_y = train_y
        self.classes = np.unique(train_y)
        self.charges = np.zeros_like(train_y, dtype=int)
        
        for _ in range(epochs):
            for i in range(self.train_x.shape[0]):
                if self.predict(self.train_x[i]) != self.train_y[i]:
                    self.charges[i] += 1

        self.train_x = self.train_x[self.charges > 0, ...]
        self.train_y = self.train_y[self.charges > 0, ...]
        self.charges = self.charges[self.charges > 0, ...]
        
    def predict(self, x: np.array):
        test_x = np.copy(x)
        
        if len(test_x.shape) < 2:
            test_x = test_x[np.newaxis, :]

        diffs = test_x[:, np.newaxis] - self.train_x[np.newaxis, :]
        assert diffs.shape[0] == test_x.shape[0] and diffs.shape[1] == self.train_x.shape[0]
        assert diffs.shape[2] == test_x.shape[1] and test_x.shape[1] == self.train_x.shape[1]
                    
        dists = np.sqrt(np.sum((diffs ** 2), axis=-1))
        assert dists.shape[0] == test_x.shape[0] and dists.shape[1] == self.train_x.shape[0]
        
        weights = self.charges * self._calculate_kernel(dists / self.window_size)
        assert weights.shape[0] == test_x.shape[0] and weights.shape[1] == self.train_x.shape[0]
        
        predicts = np.zeros((test_x.shape[0], len(self.classes)))

        for c in self.classes:
            predicts[:, c] = np.sum(weights[:, self.train_y == c], axis=1)

        return np.argmax(predicts, axis=1)
    

def plot_features(X: np.array, Y: np.array, f_names, wrongs=None):

    plt.figure(figsize=(15, 15), dpi=100)
    num_features = X.shape[1]

    for i, (f1, f2) in enumerate(product(range(num_features), range(num_features))):
        plt.subplot(num_features, num_features, 1 + i)

        if f1 == f2:
            plt.text(0.25, 0.5, f_names[f1])
        else:
            plt.scatter(X[:, f1], X[:, f2], c=Y)
            if (wrongs != None):
                plt.scatter(X[wrongs, f1], X[wrongs, f2], c='r', marker='x')

    plt.tight_layout()
    
    
def wrong_predicts(predicts, gt):
    wrong_preds =[]
    for index in range(len(predicts)):
        if predicts[index] != gt[index]:
            wrong_preds.append(index)
    return wrong_preds
