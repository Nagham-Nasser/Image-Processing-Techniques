# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:18:18 2024

@author: Nagham
"""


#1. Read an image and convert it to gray scale, then by using 
#matplotlib library then show the image and its histogram.

from matplotlib import pyplot as plt
import cv2


fig = plt.figure(figsize=(7,7))
fig2 = plt.figure(figsize=(7,7))

img = cv2.imread("0002.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(221),plt.imshow(img);
plt.subplot(222),plt.plot(gray);

plt.hist(gray.ravel(),256,[0,256]) # histogram plotting






#2. Read an image and convert it to gray scale using pillow
#library, then show the image and its histogram.
import PIL

image_pillow = PIL.Image.open(r"0002.jpg")
image_pillow_gray = image_pillow.convert('L')
histogram = image_pillow_gray.histogram()
plt.plot(histogram)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Pixel Intensity Histogram')
plt.show()



#3. Read an image and convert it to gray scale using open CV 
#library, then show the image and its histogram.



fig = plt.figure(figsize=(7,7))
fig2 = plt.figure(figsize=(7,7))

img = cv2.imread("0002.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#fig.add_subplot(2,2,1)
plt.hist(gray.ravel(),256,[0,256]) # histogram plotting
#plt.show()
#fig.add_subplot(2,2,1)
#plt.imshow(gray)




#4. Apply histogram stretching on gray image and show the 
#images after histogram stretching and its histogram.

constant = (255-0)/(img.max()-img.min())
img_stretch = img * constant

plt.hist(img_stretch.ravel(),256,[0,256])
plt.xlabel('intensity value')
plt.ylabel('number of pixels')
plt.title('Histogram of the stretched image')
plt.show ()

#5. Apply left and right histogram Sliding on gray image and 
#show the images after sliding and their histogram.
import numpy as np
def histogram_sliding(img, shift):
    value = np.ones(img.shape, np.uint8) * abs(shift)
    if shift > 0:
        return cv2.add(img, value)
    else:
        return cv2.subtract(img, value)
shift = -200
shifted_image = histogram_sliding(gray,shift)
fig2.add_subplot(2,2,1)
plt.imshow(shifted_image)


#6. Apply histogram equalization on gray image. Show the 
#original image ,the histogram of the original image , the 
#image after equalization and its histogram in the same figure. 


hist1 = cv2.calcHist([gray],[0],None,[256],[0,256])
img_2 = cv2.equalizeHist(gray)
hist2 = cv2.calcHist([img_2],[0],None,[256],[0,256])
plt.subplot(221),plt.imshow(gray);
plt.subplot(222),plt.plot(hist1);
plt.subplot(223),plt.imshow(img_2);
plt.subplot(224),plt.plot(hist2);












