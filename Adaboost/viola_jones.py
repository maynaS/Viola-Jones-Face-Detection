import numpy as np
import math
import pickle
from Feature_Extraction.image_feature_extraction import integral_image, SubMatrix
from Weak_Classifier.weak_classifier import WeakClassifier
# import selectpercentile
from sklearn.feature_selection import SelectPercentile, f_classif


class ViolaJones:
    def __init__(self, classifiers=10):
        self.classifiers = classifiers
        self.alphas = []
        self.clfs = []

    def train(self, faces, non_faces):
        training = faces + non_faces
        num_face = len(faces)
        num_non_face = len(non_faces)
        weights = np.ones(len(training))
        num_samples = len(training)
        processed_data = []
        for i in range(num_samples):
            iimg = integral_image(training[i][0])
            label = training[i][1]
            if label == 1:
                weights[i] /= (2 * num_face)
            else:
                weights[i] = 1 / (2 * num_non_face)
            processed_data.append((iimg, label))

        features = self.build_features(processed_data[0][0].shape)
        X, y = self.apply_features(features, processed_data)
        indices = SelectPercentile(f_classif, percentile=10).fit(
            X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        print("Selected %d potential features" % len(X))
        for stump in range(self.classifiers):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, processed_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" %
                  (str(clf), len(accuracy) - sum(accuracy), alpha))

    def train_weak(self, X, y, fts, weights):
        tot_pos_wts = sum(weights[y == 1])
        tot_neg_wts = sum(weights[y == 0])

        classifiers = []
        for i, ft in enumerate(X):
            # if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
            #     print("Trained %d classifiers out of %d" % (len(classifiers), total_fts))
            # sort the feature values and corresponding labels based on weights
            sorted_fts = sorted(zip(ft, y, weights), key=lambda x: x[0])
            pos = 0
            neg = 0
            pwts = 0
            nwts = 0
            mn_e, best_ft, best_thr, best_polarity = float(
                'inf'), None, None, None
            for f, wt, label in sorted_fts:
                err = min(nwts + (tot_pos_wts - pwts),
                          pwts + (tot_neg_wts - nwts))
                if err < mn_e:
                    best_polarity = 1 if pos > neg else -1
                    mn_e = err
                    best_ft = fts[i]
                    best_thr = f

                if label == 1:
                    pos += 1
                    pwts += wt
                else:
                    neg += 1
                    nwts += wt

            clf = WeakClassifier(
                best_ft[0], best_ft[1], best_thr, best_polarity)
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
                            haar_features.append(([SubMatrix(x, y, wndw, wndh)], [
                                                 SubMatrix(x+wndw, y, wndw, wndh)]))
                        if y + 2*wndh < height:
                            haar_features.append(([SubMatrix(x, y, wndw, wndh)], [
                                                 SubMatrix(x, y+wndh, wndw, wndh)]))
                        # 3 rectangles
                        if x + 3*wndw < width:
                            haar_features.append(([SubMatrix(x+wndw, y, wndw, wndh)], [
                                                 SubMatrix(x, y, wndw, wndh), SubMatrix(x+2*wndw, y, wndw, wndh)]))
                        if y + 3*wndh < height:
                            haar_features.append(([SubMatrix(x, y+wndh, wndw, wndh)], [
                                                 SubMatrix(x, y, wndw, wndh), SubMatrix(x, y+2*wndh, wndw, wndh)]))
                        # 4 rectangles
                        if x + 2*wndw < width and y + 2*wndh < height:
                            haar_features.append(([SubMatrix(x, y, wndw, wndh), SubMatrix(
                                x+wndw, y+wndh, wndw, wndh)], [SubMatrix(x+wndw, y, wndw, wndh), SubMatrix(x, y+wndh, wndw, wndh)]))
        return np.array(haar_features)

    def select_best(self, classifiers, weights, processed_data):
        """
        Selects the best weak classifier for the given weights
          Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            A tuple containing the best classifier, its error, and an array of its accuracy
        """
        bestclf = None
        besterr = float('inf')
        bestacc = None
        for clf in classifiers:
            err = 0
            acc =  []
            n = len(weights)
            assert(len(weights)==(processed_data.shape[0]))
            for i in range(n):
                ground_truth = processed_data[i][1]
                found_truth = clf.classify(processed_data[i][0])
                score = abs(found_truth - ground_truth)
                acc.append(score)
                err += weights[i] * score
            err = err / len(processed_data)
            if err < besterr:
                bestclf = clf
                besterr = err
                bestacc = acc
        return bestclf, besterr, bestacc
    
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
