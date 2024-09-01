# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:57:36 2024

@author: Nagham
"""

#  Create an array by two ways (array, NumPy) and multiply it 
#  by 2 and write the difference between the outputs.

import numpy as np 
import array as arr

x = np.array([1,2,3,4,5,6])
x=2*x
print(x)
y = arr.array('i', [1, 2, 3, 4, 5])
print(y)

#Create matrix of ones and matrix of zeros in any size.
ones = np.ones(5)
print(ones)
zero = np.zeros(3)
print(zero)

#3. Read an image and show it using OpenCV library.
import cv2
img = cv2.imread(r"0002.jpg")
print(img.shape)
print(img.dtype)
cv2.imshow('cv2 image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#4. Read an image and show it using pillow library.
from PIL import Image
img2 = Image.open(r"0002.jpg")
img2.show(img2)

#5. Read an image and show it using scikit-image library.
from skimage import io
img3 = io.imread(r"0002.jpg")
io.imshow(img3)

#6. Show image using matplotlib library. 
import matplotlib as plt
img4 = plt.pyplot.imread(r"0002.jpg")
plt.pyplot.imshow(img4)


