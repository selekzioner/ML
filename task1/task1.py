import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class PotentialFunctionClassifier:
    def __init__(self, window_size: float):
        self.window_size = window_size
        self.train_x = None
        self.train_y = None
        self.charges = None
        self.classes = None
        
        
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
        dists = np.sqrt(np.sum((diffs ** 2), axis=-1))
        
        weights = self.charges * self._calculate_kernel(dists / self.window_size)
        table = np.zeros((test_x.shape[0], len(self.classes)))

        for c in self.classes:
            table[:, c] = np.sum(weights[:, self.train_y == c], axis=1)

        return np.argmax(table, axis=1)
