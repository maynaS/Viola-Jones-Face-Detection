import numpy as np
import pickle
import time
from Adaboost.viola_jones import ViolaJones
from Cascade.cascade import CascadeClassifier

def train_helper(layers, file_name, a=-1, b=-1):
    with open("training.pkl", "rb") as f:
        training = pickle.load(f)
    if a == -1:
        clf = CascadeClassifier(layers)
    else:
        clf = CascadeClassifier(layers, a, b)
    clf.train(training)
    evaluate(clf, training)
    clf.save(file_name)

def test_helper(file_name):
    with open("test.pkl", "rb") as f:
        test = pickle.load(f)
    clf = CascadeClassifier.load(file_name)
    evaluate(clf, test)

def train_viola(t):
    train_helper(t, str(t), 2429, 4548)

def test_viola(file_name):
    test_helper(file_name)

def train_cascade(layers, file_name = "Cascade"):
    train_helper(layers, file_name, -1, -1)

def test_cascade(file_name="Cascade"):
    test_helper(file_name)


def evaluate(clf, data):
    correct = 0
    all_negs, all_pos, true_negs, false_negs, true_pos, false_pos = [0, 0, 0, 0, 0, 0]
    classif_time = 0

    for i in range(len(data)):
        x, y = data[i]
        if y == 1:
            all_pos += 1
        else:
            all_negs += 1

        start, pred = [time.time(), clf.classify(x)]
        classif_time += time.time() - start

        if pred == 1 and y == 0:
            false_pos += 1
        elif pred == 0 and y == 1:
            false_negs += 1
        
        if pred == y:
            correct += 1
        else:
            correct += 0
        
    print("False Positive Rate: {false_pos} / {all_negs} = {fpr}".format(false_pos=false_pos, all_negs=all_negs, fpr=false_pos/all_negs))
    print("False Negative Rate: {false_negs} / {all_pos} = {fpr}".format(false_negs=false_negs, all_pos=all_pos, fpr=false_negs/all_pos))
    print("Accuracy: %d / %d = %f" % (correct, len(data), correct / len(data)))
    print("Average Classification Time: %f" % (classif_time / len(data)))