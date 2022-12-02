import numpy as np


# Code to convert image to integral image
def integral_image(image):
    integral_image = np.zeros(image.shape)
    integral_image[0, 0] = image[0, 0]
    for i in range(1, image.shape[0]):
        integral_image[i, 0] = integral_image[i - 1, 0] + image[i, 0]

    for j in range(1, image.shape[1]):
        integral_image[0, j] = integral_image[0, j - 1] + image[0, j]

    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            integral_image[i, j] = integral_image[i - 1, j] + integral_image[i, j - 1] - integral_image[i - 1, j - 1] + image[i, j]

    return integral_image


# To get the sum of values in a particular submatrix of the image matrix
class SubMatrix:
    def __init__(self, y, x, width, height):
        self.x = x + 1
        self.y = y + 1
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def __str__(self):
        return "[x = {self.x}, y = {self.y}, width = {self.width}, height = {self.height}]".format(self=self)
    def __repr__(self):
        return "RectangleRegion({self.x}, {self.y}, {self.width}, {self.height})".format(self=self)

    # get sum of pixels in the submatrix [[x, y], [x + height, y + width]]
    def get_feature_val(self, img):
        A = img[self.x + self.height - 1, self.y + self.width - 1]
        B, C, D = [0, 0, 0]
        if self.x - 1 >= 0:
            B = img[self.x - 1, self.y + self.width - 1]
        if self.y - 1 >= 0:
            C = img[self.x + self.height - 1, self.y - 1]
        if self.x - 1 >= 0 and self.y - 1 >= 0:
            D = img[self.x - 1, self.y - 1]
        return A - B - C + D

# Testing the code:
# Numpy array of size [5,5]
# image = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
# print(image)
# obj = SubMatrix(1, 0, 3, 3)
# image2 = integral_image(image)
# print(image2)
# print(obj.get_feature_val(image2))

