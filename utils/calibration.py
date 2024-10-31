import numpy as np
import pandas as pd

class BBQ:
    """
    Regular Bin-Based Quantile (BBQ) calibration method.

    Parameters:
    - M: int, number of quantile-based bins.

    Methods:
    - fit(scores): Learns bin boundaries and bin means from the training data.
    - transform(scores): Calibrates scores based on the fitted bin means.
    """

    def __init__(self, M=10):
        self.M = M
        self.bin_means = None
        self.bins = None

    def fit(self, scores):
        scores = pd.Series(scores)

        # Create quantile-based bins and calculate the mean for each bin
        self.bins = pd.qcut(scores, self.M, labels=False, duplicates='drop')
        self.bin_means = scores.groupby(self.bins).mean()
        return self

    def transform(self, scores):
        scores = pd.Series(scores)
        
        # Assign each score the mean of its respective bin
        bins = pd.qcut(scores, self.M, labels=False, duplicates='drop')
        calibrated_scores = bins.map(self.bin_means).values
        return calibrated_scores

    def fit_transform(self, scores):
        self.fit(scores)
        return self.transform(scores)


class ABBQ(BBQ):
    """
    Additive Bin-Based Quantile (ABBQ) calibration method, extending BBQ.

    This method normalizes scores within each bin by scaling them relative to 
    the min and max values in each bin, with an additive term based on the bin mean.

    Methods:
    - fit(scores): Learns bin boundaries and statistics from the training data.
    - transform(scores): Calibrates scores based on the learned parameters.
    """

    def __init__(self, M=10):
        super().__init__(M)
        self.bin_min = None
        self.bin_max = None

    def fit(self, scores):
        scores = pd.Series(scores)

        # Create quantile-based bins and calculate min, max, and mean for each bin
        self.bins = pd.qcut(scores, self.M, labels=False, duplicates='drop')
        self.bin_min = scores.groupby(self.bins).min()
        self.bin_max = scores.groupby(self.bins).max()
        self.bin_means = scores.groupby(self.bins).mean()
        return self

    def transform(self, scores):
        scores = pd.Series(scores)

        # Map each score to its bin's min, max, and mean
        bins = pd.qcut(scores, self.M, labels=False, duplicates='drop')
        min_values = bins.map(self.bin_min)
        max_values = bins.map(self.bin_max)
        mean_values = bins.map(self.bin_means)

        # Normalize scores within each bin
        normalized_scores = []
        for score, min_score, max_score, mean_score in zip(scores, min_values, max_values, mean_values):
            if max_score == min_score:
                normalized_score = mean_score
            else:
                normalized_score = mean_score + (1 / self.M) * ((score - min_score) / (max_score - min_score))
            normalized_scores.append(normalized_score)

        return np.array(normalized_scores)

    def fit_transform(self, scores):
        self.fit(scores)
        return self.transform(scores)
