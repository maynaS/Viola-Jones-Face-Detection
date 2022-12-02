import numpy as np
import math
import pickle
from Feature_Extraction.image_feature_extraction import integral_image, SubMatrix
from Weak_Classifier.weak_classifier import WeakClassifier
# import selectpercentile
from sklearn.feature_selection import SelectPercentile, f_classif

class ViolaJones:
    def __init__(self, classifiers = 10):
        self.classifiers = classifiers
        self.alphas = []
        self.clfs = []

    def train(self, training, num_face, num_non_face):
        weights = np.ones(len(training))
        num_samples = len(training)
        processed_data = []
        for i in range(num_samples):
            iimg = integral_image(training[i][0])
            label = training[i][1]
            if label == 1:
                weights[i] = 1 / (2 * num_face)
            else:
                weights[i] = 1 / (2 * num_non_face)
            processed_data.append((iimg, label))
            
        features = self.build_features(processed_data[0][0].shape)
        X, y = self.apply_features(features, processed_data)
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        print("Selected %d potential features" % len(X))
        for stump in range(self.classifiers):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))

    def train_weak(self, X, y, features, weights):
        """
        Finds the optimal thresholds for each weak classifier given the current weights
          Args:
            X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            weights: A numpy array of shape len(training_data). The ith element is the weight assigned to the ith training example
          Returns:
            An array of weak classifiers
        """
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            # if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
            #     print("Trained %d classifiers out of %d" % (len(classifiers), total_features))
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers
                
    def build_features(self, img_shape):
        height = img_shape[0]
        width = img_shape[1]
        haar_features = []
        for wndw in range(1, width+1):
            for wndh in range(1, height+1):
                    for x in range(0, width-wndw+1):
                        for y in range(0, height-wndh+1):
                            # 2 rectangles
                            if x + 2*wndw < width:
                                haar_features.append(([SubMatrix(x, y, wndw, wndh)], [SubMatrix(x+wndw, y, wndw, wndh)]))
                            if y + 2*wndh < height:
                                haar_features.append(([SubMatrix(x, y, wndw, wndh)], [SubMatrix(x, y+wndh, wndw, wndh)]))
                            # 3 rectangles
                            if x + 3*wndw < width:
                                haar_features.append(([SubMatrix(x+wndw, y, wndw, wndh)], [SubMatrix(x, y, wndw, wndh), SubMatrix(x+2*wndw, y, wndw, wndh)]))
                            if y + 3*wndh < height:
                                haar_features.append(([SubMatrix(x, y+wndh, wndw, wndh)], [SubMatrix(x, y, wndw, wndh), SubMatrix(x, y+2*wndh, wndw, wndh)]))
                            # 4 rectangles
                            if x + 2*wndw < width and y + 2*wndh < height:
                                haar_features.append(([SubMatrix(x, y, wndw, wndh), SubMatrix(x+wndw, y+wndh, wndw, wndh)], [SubMatrix(x+wndw, y, wndw, wndh), SubMatrix(x, y+wndh, wndw, wndh)]))
        return np.array(haar_features)

    def select_best(self, classifiers, weights, training_data):
        """
        Selects the best weak classifier for the given weights
          Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            A tuple containing the best classifier, its error, and an array of its accuracy
        """
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy
    
    def apply_features(self, features, training_data):
        num_features = len(features)
        num_samples = len(training_data)
        X = np.zeros((num_features, num_samples))
        y = []
        for i, iimg in enumerate(training_data):
            y.append(iimg[1])
        y = np.array(y)
        i = 0
        for white_rect, black_rect in features:
            for j, iimg in enumerate(training_data):
                pos_sum = 0
                for pos in white_rect:
                    pos_sum += pos.compute_feature(iimg)
                neg_sum = 0
                for neg in black_rect:
                    neg_sum += neg.compute_feature(iimg)
                X[i][j] = pos_sum - neg_sum            
            i += 1
        return X, y

    def classify(self, image):
        """
        Classifies an image
          Args:
            image: A numpy 2D array of shape (m, n) representing the image
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0