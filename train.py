import os
import PIL.Image
import numpy as np
from Feature_Extraction.image_feature_extraction import integral_image
from Cascade.cascade import CascadeClassifier


def load_images(path):
    images = []
    cnt = 0
    for _file in os.listdir(path):
        if _file.endswith('.pgm') and cnt < 400:
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

# print(faces_ii_training[0].shape)
# print(faces_ii_training[0])
# print(non_faces_ii_training[0].shape)
# print(non_faces_ii_training[0])
print('Training classifier..')
classifier = CascadeClassifier([3, 10, 20, 50])
classifier.train(faces_ii_training+non_faces_ii_training)
print('..done.\n')
classifier.save('classifier_1_3')