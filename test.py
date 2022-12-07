import os
import PIL.Image
import numpy as np
from Feature_Extraction.image_feature_extraction import integral_image
from Cascade.cascade import CascadeClassifier
from sklearn.metrics import *
from typing import *
import matplotlib.pyplot as plt
import seaborn as sns

def load_images(path):
    images = []
    cnt = 0
    for _file in os.listdir(path):
        if _file.endswith('.pgm') and cnt < 1000:
            img_arr = np.array(PIL.Image.open(
                (os.path.join(path, _file))), dtype=np.float64)
            if img_arr.max() != 0:
                img_arr /= img_arr.max()
            images.append(img_arr)
            cnt += 1
    return images

pos_testing_path = 'test/face'
neg_testing_path = 'test/non-face'

faces_testing = load_images(pos_testing_path)
faces_ii_testing = list(map(integral_image, faces_testing))
faces_ii_testing = list(zip(faces_ii_testing, [1] * len(faces_ii_testing)))
non_faces_testing = load_images(neg_testing_path)
non_faces_ii_testing = list(map(integral_image, non_faces_testing))
non_faces_ii_testing = list(zip(non_faces_ii_testing, [0] * len(non_faces_ii_testing)))
print("Faces: " + str(len(faces_ii_testing)) + " Non faces: " + str(len(non_faces_ii_testing)))

classifier = CascadeClassifier.load('classifier_final')

score1 = 0
score2 = 0
y_tru = []
y_pre = []
for ex in faces_ii_testing:
    y_tru.append(1)
    if classifier.classify(ex[0]) == 1:
        y_pre.append(1)
        score1 += 1
    elif classifier.classify(ex[0]) == 0:
        y_pre.append(0)
for ex in non_faces_ii_testing:
    y_tru.append(0)
    if classifier.classify(ex[0]) == 0:
        y_pre.append(0)
        score2 += 1
    elif classifier.classify(ex[0]) == 1:
        y_pre.append(1)
print('..done. Score: ' + str(score1) + '/' + str(len(faces_ii_testing)) +
        ' faces and ' + str(score2) + '/' + str(len(non_faces_ii_testing)) + ' non faces.')
print('..done.\n')

PredictionStats = NamedTuple('PredictionStats', [('tn', int), ('fp', int), ('fn', int), ('tp', int)])

def prediction_stats(y_true, y_pred):
    c = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = c.ravel()
    return c, PredictionStats(tn=tn, fp=fp, fn=fn, tp=tp)

c, s = prediction_stats(y_tru, y_pre)

sns.heatmap(c / c.sum(), cmap='YlGnBu', annot=True, square=True, fmt='.1%',
            xticklabels=['Predicted negative', 'Predicted positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion matrix for the strong classifier');

print(f'Precision {s.tp/(s.tp+s.fp):.2f}, recall {s.tp/(s.tp+s.fn):.2f}, false positive rate {s.fp/(s.fp+s.tn):.2f}, false negative rate {s.fn/(s.tp+s.fn):.2f}.')
plt.savefig('Conf_Mat.png')