# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:23:10 2024

@author: Nagham
"""

# 1.By using open cv library read an image and apply the
# following filters:
import cv2
image = cv2.imread(r"0002.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# bilateralFilter
bilateralFilter = cv2.bilateralFilter(gray,9,75,75)
cv2.imshow('bilateralFilter', bilateralFilter)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows() 
# medianBlur
medianBlur = cv2.medianBlur(gray, 9)
cv2.imshow('medianBlur', medianBlur)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows() 
# GaussianBlur
GaussianBlur = cv2.GaussianBlur(gray,(9,9),21)
cv2.imshow('GaussianBlur', GaussianBlur)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()
# blur
blur = cv2.blur(gray,(3,3))
cv2.imshow('blur', blur)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()
# cv2.boxFilter
boxFilter = cv2.boxFilter(gray, -1, (3,3))
cv2.imshow('boxFilter', boxFilter)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()
# laplacian
laplacian = cv2.Laplacian(gray, -1)
cv2.imshow('laplacian', laplacian)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()
# 2-By Using cv2.filter2D() and numpy apply any kernel
# to the image.
import cv2
import numpy as np

kernel = np.array([[2, 1, 0],
                   [-1, 1, 1],
                   [0, 1, -2]])

# Apply the kernel using cv2.filter2D()
filtered_image = cv2.filter2D(gray, -1, kernel)

# Display the filtered image
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()  # Close all OpenCV windows


# 3- using the cv2.getGaussianKernel().apply the output
# gaussian kernel to the image.
gaus = cv2.getGaussianKernel(3, 0.5)
filtered_image = cv2.filter2D(gray, -1, gaus)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()  
