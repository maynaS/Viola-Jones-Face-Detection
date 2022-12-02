import numpy as np


# Weak classifier
class WeakClassifier:
    def __init__(self, positive_region, negative_region, threshold, polarity):
        self.positive_region = positive_region
        self.negative_region = negative_region
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        feature = lambda ii: sum([pos.get_feature_val(ii) for pos in self.positive_regions]) - sum([neg.get_feature_val(ii) for neg in self.negative_regions])
        if self.polarity * feature(x) < self.polarity * self.threshold:
            return 1
        return 0
    