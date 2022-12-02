import numpy as np


# Weak classifier
class WeakClassifier:
    def __init__(self, positive_region, negative_region, threshold, polarity):
        self.positive_region = positive_region
        self.negative_region = negative_region
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        pos_sum = 0
        for pos in self.positive_region:
            pos_sum += pos.compute_feature(x)
        neg_sum = 0
        for neg in self.negative_region:
            neg_sum += neg.compute_feature(x)
        return 1 if self.polarity * (pos_sum - neg_sum) < self.polarity * self.threshold else 0
    