# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:29:41 2024

@author: Nagham
"""

#1. By using open cv library read colored image, convert it to gray image and save it.
import cv2
image1 = cv2.imread(r"0002.jpg")
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg", gray)

#2. By using pillow library read colored image, convert it to gray image and save it.
import PIL
image2 = PIL.Image.open(r"0002.jpg")
gray2 = image2.convert("L")
gray2.save("gray2.jpg")

# 3. Read a gray image and apply the log transformation on the 
# image. show the original and transformation images.

import numpy as np
c = 255 / np.log(1+np.max(gray))
log_img = c * np.log(1+gray)

# 4. Read an image in gray scale level and apply the Power-law 
#(gamma) transformation with different gamma [0.1, 0.5, 1.2, 
# 2.2] on the image. show the original and transformation 
# images in the same figure.
from matplotlib import pyplot as plt

gamma=0.1
gama_img = 255 * ((gray/255)**gamma)
gama_img = np.array(gama_img,dtype=np.uint8)
plt.imshow(gama_img,cmap="gray")




















