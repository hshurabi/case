import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class TSKFold:
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, time_column, event_column):
        """
        Split the data into stratified k-folds based on time and class ratio.

        Parameters:
        - X: DataFrame, features
        - y: array-like, target variable with event indicators
        - time_column: str, column name representing the time period
        - event_column: str, column name representing the event indicator

        Yields:
        - train_index, test_index: indices for train/test splits
        """
        # Combine X and y for stratification
        data = X.copy()
        data['time'] = y[time_column]
        data['event'] = y[event_column]

        # Create an empty list to hold fold indices
        fold_indices = [[] for _ in range(self.n_splits)]

        # Group by time periods
        for time, group in data.groupby('time'):
            # Perform stratified k-fold on each time period to maintain class ratio
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            for fold, (train_idx, test_idx) in enumerate(skf.split(group, group['event'])):
                fold_indices[fold].append(group.index[test_idx].tolist())

        # Flatten indices for each fold and yield them
        for fold in range(self.n_splits):
            test_indices = np.concatenate(fold_indices[fold])
            train_indices = data.index.difference(test_indices)
            yield train_indices, test_indices
