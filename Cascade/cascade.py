import pickle
from Adaboost.viola_jones import ViolaJones

class CascadeClassifier:
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []

    def train(self, training):
        positive = []
        negative = []

        for i in range(len(training)):
            if training[i][1] == 1:
                positive.append(training[i])
            else:
                negative.append(training[i])
        print(negative.__len__())
        for i in self.layers:
            # if len(negative) == 0:
            #     print("Stopping early")
            #     break
            
            classifier = ViolaJones(classifiers = i)
            classifier.train(positive, negative)
            self.clfs.append(classifier)
            false_positives = []
            for i in range(len(negative)):
                if self.classify(negative[i][0]) == 1:
                    false_positives.append(negative[i])

            negative = false_positives
    
    def classify(self, img):
        for classifier in self.clfs:
            if classifier.classify(img) == 0:
                return 0
        return 1

    def save(self, file_name):
        with open(file_name + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name + ".pkl", 'rb') as f:
            return pickle.load(f)