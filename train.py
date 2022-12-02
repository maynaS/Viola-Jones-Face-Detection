import os
import PIL.Image
import numpy as np
from Feature_Extraction.image_feature_extraction import integral_image
from Cascade.cascade import CascadeClassifier


def load_images(path):
    images = []
    cnt = 0
    for _file in os.listdir(path):
        if _file.endswith('.pgm') and cnt < 100:
            img_arr = np.array(PIL.Image.open(
                (os.path.join(path, _file))), dtype=np.float64)
            if img_arr.max() != 0:
                img_arr /= img_arr.max()
            images.append(img_arr)
            cnt += 1
    return images


pos_training_path = 'train/face'
neg_training_path = 'train/non-face'
pos_testing_path = 'test/face'
neg_testing_path = 'test/non-face'

print('Loading faces for training..')
faces_training = load_images(pos_training_path)
faces_ii_training = list(map(integral_image, faces_training))
faces_ii_training = list(zip(faces_ii_training, [1] * len(faces_ii_training)))
print('..done. ' + str(len(faces_training)) +
      ' faces loaded.\n\nLoading non faces..')
non_faces_training = load_images(neg_training_path)
non_faces_ii_training = list(map(integral_image, non_faces_training))
non_faces_ii_training = list(zip(non_faces_ii_training, [0] * len(non_faces_ii_training)))
print('..done. ' + str(len(non_faces_training)) + ' non faces loaded.\n')
faces_testing = load_images(pos_testing_path)
faces_ii_testing = list(map(integral_image, faces_testing))
faces_ii_testing = list(zip(faces_ii_testing, [1] * len(faces_ii_testing)))
print('..done. ' + str(len(faces_testing)) +
      ' faces loaded.\n\nLoading test non faces..')
non_faces_testing = load_images(neg_testing_path)
non_faces_ii_testing = list(map(integral_image, non_faces_testing))
non_faces_ii_testing = list(zip(non_faces_ii_testing, [0] * len(non_faces_ii_testing)))
print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

# print(faces_ii_training[0].shape)
# print(faces_ii_training[0])
# print(non_faces_ii_training[0].shape)
# print(non_faces_ii_training[0])
print('Training classifier..')
classifier = CascadeClassifier(5)
classifier.train(faces_ii_training+non_faces_ii_training)
print('..done.\n')

score1 = 0
score2 = 0
for ex in faces_ii_testing:
    if classifier.classify(ex[0]) == 1:
        score1 += 1
for ex in non_faces_ii_testing:
    if classifier.classify(ex[0]) == 0:
        score2 += 1
print('..done. Score: ' + str(score1) + '/' + str(len(faces_ii_testing)) +
        ' faces and ' + str(score2) + '/' + str(len(non_faces_ii_testing)) + ' non faces.')
print('..done.\n')